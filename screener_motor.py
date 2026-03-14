import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import requests
from typing import List, Dict, Any, Tuple

# User-Agent for Wikipedia-forespørsler (pd.read_html uten dette blokkeres ofte)
WIKIPEDIA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ============================================================
# KONSTANTER / PARAMETRE
# ============================================================

MARKET_INDEX_TICKER = "SPY"

TRAIN_RATIO        = 0.70
MIN_TEST_DAYS      = 60
TIME_PERIOD_YEARS  = 2
SLEEP_TIME_SECONDS = 0.3   # Per batch (ikke per ticker)

MIN_CANDIDATES_FOR_SECTOR_RANK = 20
MASTER_LIST_CACHE = "sp1500_master_cache.csv"

# ---- Univers: S&P 1500 (S&P 500 + S&P 400 + S&P 600) ----
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
SP600_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"

# ---- Pre-filter i Pass 1 ----
MIN_PRICE_USD      = 5.0          # Minimum kurs (USD)
MIN_DOLLAR_VOL_USD = 1_000_000    # Minimum daglig omsetning 50d-snitt (USD)

# ---- Batch-nedlasting i Pass 1 ----
BATCH_SIZE = 100

# ---- ATR ----
ATR_PERIOD        = 14
ATR_SL_MULTIPLIER = 2.0   # Dynamisk SL = siste kurs − 2 × ATR(14)

# ---- GATE: eneste absolutte krav ----
SMA50_CROSS_LOOKBACK_DAYS = 10   # ca. 2 uker (handelsdager)

# ---- BONUSPOENG ----
SMA150_BONUS_POINTS       = 10.0   # Kryss opp over SMA150 nylig + fortsatt over
GOLDEN_CROSS_BONUS_POINTS =  8.0   # MA50 > MA200 (Gull-kryss)
VOL_CONFIRM_CROSS_BONUS   =  5.0   # Høyt volum på selve SMA50-crossover-dagen

# ---- VEKTING AV SCORE ----
WEIGHTS = {
    "vb_percentile":  0.50,
    "mom_percentile": 0.30,
    "vol_percentile": 0.20,
}

# ============================================================
# MASTERLISTE – S&P 1500 via Wikipedia
# ============================================================

def _scrape_index_table(url: str) -> pd.DataFrame:
    """
    Forsøker å hente konstituenttabell fra en Wikipedia-side.
    Bruker requests med User-Agent for å unngå blokkering (GitHub Actions IP-er).
    Returnerer DataFrame med kolonner [ticker, name, sector], eller tom DataFrame.
    """
    try:
        response = requests.get(url, headers=WIKIPEDIA_HEADERS, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(response.text)
    except Exception as e:
        print(f"  Feil ved henting fra {url}: {e}")
        return pd.DataFrame()

    for tbl in tables:
        cols = [str(c).strip() for c in tbl.columns]

        ticker_col = next(
            (c for c in cols if c in ("Symbol", "Ticker symbol", "Ticker")), None
        )
        if ticker_col is None:
            continue

        name_col = next(
            (c for c in cols if any(k in c for k in ("Security", "Company", "Name"))), None
        )
        sector_col = next(
            (c for c in cols if any(k in c for k in ("Sector", "GICS"))), None
        )

        result = pd.DataFrame()
        result["ticker"] = (
            tbl[ticker_col].astype(str).str.strip().str.replace(".", "-", regex=False)
        )
        result["name"]   = tbl[name_col].astype(str).str.strip() if name_col else ""
        result["sector"] = tbl[sector_col].astype(str).str.strip() if sector_col else "Unknown"

        result = result.dropna(subset=["ticker"])
        result = result[result["ticker"].str.match(r"^[A-Z]{1,5}(-[A-Z])?$", na=False)]

        if not result.empty:
            return result.reset_index(drop=True)

    return pd.DataFrame()


def get_master_ticker_list() -> pd.DataFrame:
    """
    Henter S&P 1500 (S&P 500 + S&P 400 + S&P 600) fra Wikipedia.
    Fallback til cachet CSV-fil, deretter bare S&P 500.
    """
    print("-> Starter henting av masterliste (S&P 1500).")

    frames = []
    for label, url in [("S&P 500", SP500_URL), ("S&P 400", SP400_URL), ("S&P 600", SP600_URL)]:
        df = _scrape_index_table(url)
        if not df.empty:
            frames.append(df)
            print(f"   {label}: {len(df)} aksjer.")
        else:
            print(f"   {label}: ingen data – hopper over.")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset="ticker").reset_index(drop=True)
        combined.to_csv(MASTER_LIST_CACHE, index=False)
        print(f"-> Totalt {len(combined)} unike tickers cachet.")
        return combined

    if os.path.exists(MASTER_LIST_CACHE):
        df = pd.read_csv(MASTER_LIST_CACHE)
        print(f"-> Liste hentet fra cache ({len(df)} tickers).")
        return df

    print("-> Fallback: prøver bare S&P 500 med requests.")
    try:
        response = requests.get(SP500_URL, headers=WIKIPEDIA_HEADERS, timeout=30)
        response.raise_for_status()
        sp500_df = pd.read_html(response.text)[0]
        df = sp500_df[["Symbol", "Security", "GICS Sector"]].rename(
            columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"}
        )
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        print(f"-> Fallback S&P 500: {len(df)} tickers.")
        return df
    except Exception as e:
        print(f"-> KRITISK FEIL: Ingen masterliste funnet. Detaljer: {e}")
        return pd.DataFrame(columns=["ticker", "name", "sector"])


# ============================================================
# HJELPEFUNKSJONER (batch + ATR)
# ============================================================

def _get_ticker_df(batch_df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Trekker ut enkelt-ticker DataFrame fra batch yf.download (group_by='ticker')."""
    try:
        if isinstance(batch_df.columns, pd.MultiIndex):
            outer = batch_df.columns.get_level_values(0)
            if ticker not in outer:
                return None
            df = batch_df[ticker].copy()
        else:
            df = batch_df.copy()
        df = df.dropna(how="all")
        return df if not df.empty else None
    except (KeyError, TypeError):
        return None


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD
) -> pd.Series:
    """Average True Range med Wilder's smoothing (EMA alpha = 1/period)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _days_since_last_true(bool_series: pd.Series) -> Any:
    """Returnerer antall dager siden siste True i en bool-serie (0 = i dag)."""
    if bool_series is None or bool_series.empty or not bool_series.any():
        return None
    last_true_pos = np.where(bool_series.values)[0][-1]
    return int(len(bool_series) - 1 - last_true_pos)


# ============================================================
# PASS 1: Teknisk metrikk (gate + bonus-flagg + ATR)
# ============================================================

def calculate_technical_metrics(df: pd.DataFrame, ticker: str) -> Dict[str, Any] | None:
    """
    Beregner tekniske metrikker og gate/bonus-signaler.

    Gate (begge må være true):
      - Kryss opp over SMA50 innen siste N handelsdager
      - Fortsatt over SMA50 i dag

    Bonus-signaler (gir ekstrapoeng i Pass 3):
      1. SMA150-kryss + fortsatt over SMA150
      2. Golden Cross: SMA50 > SMA200
      3. Volum-bekreftet crossover: volum på crossover-dag >= 1.5 × 20d-snitt

    Returnerer også ATR(14) og dynamisk stop-loss-prosent.
    """
    if df is None or df.empty:
        return None

    if df.shape[0] < 200:
        return None

    required_cols = {"Close", "High", "Low", "Open", "Volume"}
    if not required_cols.issubset(set(df.columns)):
        return None

    df = df.copy()

    # Glidende snitt
    df["SMA50"]  = df["Close"].rolling(window=50).mean()
    df["SMA150"] = df["Close"].rolling(window=150).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()

    # Momentum
    df["MomentumRatio"] = df["Close"] / df["SMA50"] - 1

    # Volum / likviditet
    df["DollarVolume"]   = df["Close"] * df["Volume"]
    df["AvgDollarVol5"]  = df["DollarVolume"].rolling(window=5).mean()
    df["AvgDollarVol50"] = df["DollarVolume"].rolling(window=50).mean()
    df["VolRatio"]       = df["AvgDollarVol5"] / df["AvgDollarVol50"] - 1

    # ATR
    df["ATR14"] = calculate_atr(df["High"], df["Low"], df["Close"], period=ATR_PERIOD)

    last_row = df.iloc[-1]
    for col in ["SMA50", "SMA150", "SMA200", "AvgDollarVol50", "ATR14"]:
        if pd.isna(last_row[col]):
            return None

    # --- GATE: SMA50 cross-up ---
    cross50_series  = (df["Close"] > df["SMA50"]) & (df["Close"].shift(1) <= df["SMA50"].shift(1))
    cross50_last_n  = bool(cross50_series.tail(SMA50_CROSS_LOOKBACK_DAYS).any())
    still_above_50  = bool(df["Close"].iloc[-1] > df["SMA50"].iloc[-1])
    days_since_cross50 = _days_since_last_true(cross50_series)

    # --- BONUS 1: SMA150 cross-up ---
    cross150_series    = (df["Close"] > df["SMA150"]) & (df["Close"].shift(1) <= df["SMA150"].shift(1))
    cross150_recent    = bool(cross150_series.tail(SMA50_CROSS_LOOKBACK_DAYS).any())
    still_above_150    = bool(df["Close"].iloc[-1] > df["SMA150"].iloc[-1])
    days_since_cross150 = _days_since_last_true(cross150_series)

    # --- BONUS 2: Golden Cross (SMA50 > SMA200) ---
    golden_cross = bool(last_row["SMA50"] > last_row["SMA200"])

    # --- BONUS 3: Volum-bekreftet crossover ---
    vol_confirmed_cross50 = False
    if cross50_last_n:
        cross_dates = cross50_series[cross50_series].index
        if len(cross_dates) > 0:
            last_cross_date  = cross_dates[-1]
            avg_vol_20       = df["Volume"].rolling(20).mean()
            cross_day_vol    = df.loc[last_cross_date, "Volume"]
            cross_day_avg    = avg_vol_20.loc[last_cross_date]
            if not pd.isna(cross_day_avg) and cross_day_avg > 0:
                vol_confirmed_cross50 = bool(cross_day_vol >= 1.5 * cross_day_avg)

    # ATR-basert dynamisk stop-loss
    last_close      = float(last_row["Close"])
    last_atr        = float(last_row["ATR14"])
    dynamic_sl_pct  = (ATR_SL_MULTIPLIER * last_atr / last_close) * 100  # % av kurs

    return {
        "ticker":    ticker,
        "asof_date": df.index[-1].strftime("%Y-%m-%d"),
        "last_close": last_close,

        "momentum_ratio":    float(last_row["MomentumRatio"]),
        "vol_ratio":         float(last_row["VolRatio"]),
        "avg_dollar_vol_50": float(last_row["AvgDollarVol50"]),

        "cross50_last_n":     cross50_last_n,
        "still_above_50":     still_above_50,
        "days_since_cross50": days_since_cross50,

        "cross150_recent":     cross150_recent,
        "still_above_150":     still_above_150,
        "days_since_cross150": days_since_cross150,

        "golden_cross":          golden_cross,
        "vol_confirmed_cross50": vol_confirmed_cross50,

        "atr_14":        last_atr,
        "dynamic_sl_pct": dynamic_sl_pct,
    }


# ============================================================
# PASS 2: V/B (robust simulering)
# ============================================================

def simuler_handel_for_sim(df, kjops_dato, kjops_pris, stop_loss_pct):
    """Simulerer Trailing Stop Loss handel og returnerer kumulativ avkastning."""
    periode_data = df[df.index >= kjops_dato].copy()
    if periode_data.empty:
        return pd.Series(dtype=float)

    hoyeste_pris  = kjops_pris
    posisjon_verdi = pd.Series(index=periode_data.index, dtype=float)
    exit_dato      = None

    posisjon_verdi.iloc[0] = kjops_pris

    for i, (dato, row) in enumerate(periode_data.iterrows()):
        if row["High"] > hoyeste_pris:
            hoyeste_pris = row["High"]

        stop_niva = hoyeste_pris * (1 - stop_loss_pct)

        if row["Low"] <= stop_niva:
            posisjon_verdi.iloc[i:] = stop_niva
            exit_dato = dato
            break

        posisjon_verdi.iloc[i] = row["Close"]

    if exit_dato is None:
        posisjon_verdi.iloc[i:] = periode_data.iloc[-1]["Close"]

    return (posisjon_verdi / kjops_pris) - 1


def calculate_cagr(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    total_return = returns.iloc[-1] + 1.0
    antall_dager = (returns.index[-1] - returns.index[0]).days
    if antall_dager <= 0:
        return 0.0
    return (total_return ** (1 / (antall_dager / 365.25))) - 1.0


def calculate_max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity  = returns + 1.0
    peak    = equity.cummax()
    dd      = (equity / peak) - 1
    return abs(dd.min())


def pick_start_dates(df: pd.DataFrame, min_lookback: int = 60, step: int = 10) -> List[pd.Timestamp]:
    if len(df) <= min_lookback + 5:
        return []
    return list(df.index[min_lookback::step])


def grid_search_sl_multi_start(
    df: pd.DataFrame,
    sl_range: range,
    start_dates: List[pd.Timestamp],
    use_percentile: int = 50,
) -> Tuple[float | None, pd.DataFrame]:
    """Stop-loss grid-søk på flere startdatoer (50-persentil median)."""
    if not start_dates:
        return None, pd.DataFrame()

    results = []
    for sl in sl_range:
        sl_dec = sl / 100.0
        per_start = []
        for start in start_dates:
            if start not in df.index:
                continue
            buy_price = float(df.loc[start, "Close"])
            ret = simuler_handel_for_sim(df, start, buy_price, sl_dec)
            if not ret.empty:
                per_start.append(ret.iloc[-1])
        if not per_start:
            continue
        results.append({"stop_loss_pct": sl, "score": float(np.percentile(per_start, use_percentile))})

    res_df = pd.DataFrame(results).sort_values("score", ascending=False)
    if res_df.empty:
        return None, res_df

    return res_df.iloc[0]["stop_loss_pct"] / 100.0, res_df


def run_robust_vb_simulation(df_full: pd.DataFrame) -> Dict[str, Any] | None:
    """Train/Test split → optimal SL på train → CAGR og MaxDD på test → V/B ratio."""
    if df_full is None or df_full.empty or df_full.shape[0] < 200:
        return None

    split_idx  = int(df_full.shape[0] * TRAIN_RATIO)
    df_train   = df_full.iloc[:split_idx].copy()
    df_test    = df_full.iloc[split_idx:].copy()

    if df_test.shape[0] < MIN_TEST_DAYS:
        return None

    train_starts = pick_start_dates(df_train, min_lookback=60, step=10)
    best_sl, _   = grid_search_sl_multi_start(df_train, range(3, 81), train_starts, use_percentile=50)

    if best_sl is None:
        return None

    kjops_dato  = df_test.index[0]
    kjops_pris  = float(df_test.iloc[0]["Open"])
    test_ret    = simuler_handel_for_sim(df_test, kjops_dato, kjops_pris, best_sl)

    if test_ret.empty:
        return None

    cagr_test  = calculate_cagr(test_ret)
    max_dd     = calculate_max_drawdown(test_ret)
    vb_ratio   = 0.0 if cagr_test <= 0 else cagr_test / max(max_dd, 0.05)

    return {
        "vb_ratio":             vb_ratio,
        "optimal_sl_train":     best_sl * 100,
        "cagr_test_percent":    cagr_test * 100,
        "max_drawdown_test":    max_dd * 100,
    }


# ============================================================
# HOVEDSCREENER
# ============================================================

def run_full_screener():
    # 0) Pass 0: Regime Check (kun informativt)
    print("\n--- 0. Pass 0: Regime Check ---")
    try:
        idx_data         = yf.download(MARKET_INDEX_TICKER, period="2y", interval="1d", progress=False)
        idx_data["SMA200"] = idx_data["Close"].rolling(200).mean()
        idx_close        = float(idx_data["Close"].iloc[-1])
        idx_sma200       = float(idx_data["SMA200"].iloc[-1])

        if pd.isna(idx_sma200) or idx_close < idx_sma200:
            print("MARKNEDSREGIME: Bearish/Ubestemt. Fortsetter screening likevel.")
        else:
            print("MARKNEDSREGIME: Bullish.")
    except Exception as e:
        print(f"Advarsel: Kunne ikke sjekke regime. Fortsetter. Detaljer: {e}")

    # 1) Masterliste
    ticker_list_df = get_master_ticker_list()
    print(f"\nMasterliste: {len(ticker_list_df)} tickers lastet.")
    if ticker_list_df.empty:
        print("KRITISK: Ingen tickers funnet. Avslutter.")
        return pd.DataFrame()

    tickers     = ticker_list_df["ticker"].tolist()
    ticker_meta = ticker_list_df.set_index("ticker").to_dict("index")
    total_tickers = len(tickers)
    print(f"\nStarter screening av {total_tickers} aksjer (S&P 1500).")
    print(f"Eksempel-tickers: {tickers[:5]}")

    # 2) Pass 1: Gate på SMA50-cross (batch-nedlasting)
    total_batches = (total_tickers + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\n--- 1. Pass 1: Gate (SMA50-cross) – {total_batches} batches à {BATCH_SIZE} ---")
    shortlist_data = []

    for b_start in range(0, total_tickers, BATCH_SIZE):
        batch      = tickers[b_start : b_start + BATCH_SIZE]
        batch_num  = b_start // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{total_batches}: {len(batch)} tickers ...", end=" ", flush=True)

        try:
            batch_df = yf.download(
                batch, period="1y", interval="1d",
                progress=False, group_by="ticker", threads=True,
            )
        except Exception as e:
            print(f"FEIL: {e}")
            time.sleep(SLEEP_TIME_SECONDS)
            continue

        passed = 0
        for ticker in batch:
            try:
                df = _get_ticker_df(batch_df, ticker)
                if df is None or len(df) < 200:
                    continue

                # Pre-filter: pris og likviditet
                last_close = df["Close"].iloc[-1]
                if pd.isna(last_close) or last_close < MIN_PRICE_USD:
                    continue

                dollar_vol_50 = (df["Close"] * df["Volume"]).rolling(50).mean().iloc[-1]
                if pd.isna(dollar_vol_50) or dollar_vol_50 < MIN_DOLLAR_VOL_USD:
                    continue

                metrics = calculate_technical_metrics(df, ticker)
                if metrics is None:
                    continue
                if not metrics.get("cross50_last_n", False):
                    continue
                if not metrics.get("still_above_50", False):
                    continue

                meta = ticker_meta.get(ticker, {})
                shortlist_data.append({
                    "ticker": ticker,
                    "name":   meta.get("name", ""),
                    "sector": meta.get("sector", "Unknown"),
                    **metrics,
                })
                passed += 1

            except Exception:
                continue

        print(f"{passed} passerte.")
        time.sleep(SLEEP_TIME_SECONDS)

    shortlist_df = pd.DataFrame(shortlist_data)
    if shortlist_df.empty:
        print("Ingen aksjer kvalifisert etter Pass 1 (SMA50-cross gate).")
        return pd.DataFrame()

    print(f"Pass 1 fullført. {len(shortlist_df)} aksjer kvalifisert for Pass 2.")

    # 3) Pass 2: V/B simulering (individuelle 2-års nedlastninger)
    print("\n--- 2. Pass 2: Dyp Analyse (V/B Simulering) ---")
    final_candidates = []

    for _, row in shortlist_df.iterrows():
        ticker = row["ticker"]
        time.sleep(SLEEP_TIME_SECONDS)

        try:
            data       = yf.download(ticker, period=f"{TIME_PERIOD_YEARS}y", interval="1d", progress=False)
            vb_results = run_robust_vb_simulation(data)
            if vb_results:
                final_candidates.append({**row.to_dict(), **vb_results})
        except Exception:
            continue

    final_df = pd.DataFrame(final_candidates)
    if final_df.empty:
        print("Ingen aksjer kvalifisert etter Pass 2.")
        return pd.DataFrame()

    # 4) Pass 3: Normalisering + score
    print("\n--- 3. Pass 3: Normalisering og Scoring ---")

    def robust_normalize(df: pd.DataFrame, col: str) -> np.ndarray:
        sector_ranks  = df.groupby("sector")[col].transform(lambda x: x.rank(pct=True) * 100)
        sector_counts = df.groupby("sector")[col].transform("count")
        global_rank   = df[col].rank(pct=True) * 100
        return np.where(sector_counts >= MIN_CANDIDATES_FOR_SECTOR_RANK, sector_ranks, global_rank)

    final_df["vb_percentile"]  = robust_normalize(final_df, "vb_ratio")
    final_df["mom_percentile"] = robust_normalize(final_df, "momentum_ratio")

    q_low  = final_df["vol_ratio"].quantile(0.01)
    q_high = final_df["vol_ratio"].quantile(0.99)
    final_df["vol_ratio_capped"] = final_df["vol_ratio"].clip(lower=q_low, upper=q_high)
    final_df["vol_percentile"]   = robust_normalize(final_df, "vol_ratio_capped")

    final_df["total_score"] = (
        final_df["vb_percentile"]  * WEIGHTS["vb_percentile"]
        + final_df["mom_percentile"] * WEIGHTS["mom_percentile"]
        + final_df["vol_percentile"] * WEIGHTS["vol_percentile"]
    )

    # Bonus 1: SMA150
    final_df["sma150_bonus"] = 0.0
    mask150 = (
        (final_df.get("cross150_recent", False) == True)
        & (final_df.get("still_above_150", False) == True)
    )
    final_df.loc[mask150, "sma150_bonus"] = SMA150_BONUS_POINTS

    # Bonus 2: Golden Cross
    final_df["golden_cross_bonus"] = 0.0
    mask_gc = final_df.get("golden_cross", False) == True
    final_df.loc[mask_gc, "golden_cross_bonus"] = GOLDEN_CROSS_BONUS_POINTS

    # Bonus 3: Volum-bekreftet crossover
    final_df["vol_confirm_bonus"] = 0.0
    mask_vc = final_df.get("vol_confirmed_cross50", False) == True
    final_df.loc[mask_vc, "vol_confirm_bonus"] = VOL_CONFIRM_CROSS_BONUS

    final_df["total_score"] = (
        final_df["total_score"]
        + final_df["sma150_bonus"]
        + final_df["golden_cross_bonus"]
        + final_df["vol_confirm_bonus"]
    )

    final_df = final_df.sort_values("total_score", ascending=False).head(50).reset_index(drop=True)
    final_df["run_id"] = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

    def _fmt(x, default="n/a"):
        try:
            return default if pd.isna(x) else x
        except Exception:
            return default

    final_df["why_selected"] = final_df.apply(
        lambda x: (
            f"Cross50: {int(_fmt(x.get('days_since_cross50'), 999))}d siden"
            f" | GoldenX: {bool(_fmt(x.get('golden_cross'), False))}"
            f" | VolConf: {bool(_fmt(x.get('vol_confirmed_cross50'), False))}"
            f" | SMA150+: {bool(_fmt(x.get('cross150_recent'), False) and _fmt(x.get('still_above_150'), False))}"
            f" | V/B pct: {x['vb_percentile']:.1f}"
            f" | Mom: {x.get('momentum_ratio', 0.0) * 100:.1f}%"
            f" | SL(VB): {_fmt(x.get('optimal_sl_train', np.nan), np.nan):.1f}%"
            f" | ATR-SL: {_fmt(x.get('dynamic_sl_pct', np.nan), np.nan):.1f}%"
        ),
        axis=1,
    )

    output_cols = [
        "asof_date", "run_id",
        "ticker", "name", "sector",
        "total_score", "vb_percentile", "mom_percentile", "vol_percentile",
        "sma150_bonus", "golden_cross_bonus", "vol_confirm_bonus",
        "golden_cross", "vol_confirmed_cross50",
        "optimal_sl_train", "cagr_test_percent", "max_drawdown_test",
        "atr_14", "dynamic_sl_pct",
        "avg_dollar_vol_50",
        "why_selected",
    ]

    for c in output_cols:
        if c not in final_df.columns:
            final_df[c] = np.nan

    return final_df[output_cols]


# ============================================================
# EKSPORT
# ============================================================

def export_results(df: pd.DataFrame):
    """
    Lagrer latest CSV og en datostemplet CSV.
    Skriver ALLTID latest (også når df er tom), så Streamlit aldri får 404.
    """
    expected_cols = [
        "asof_date", "run_id",
        "ticker", "name", "sector",
        "total_score", "vb_percentile", "mom_percentile", "vol_percentile",
        "sma150_bonus", "golden_cross_bonus", "vol_confirm_bonus",
        "golden_cross", "vol_confirmed_cross50",
        "optimal_sl_train", "cagr_test_percent", "max_drawdown_test",
        "atr_14", "dynamic_sl_pct",
        "avg_dollar_vol_50",
        "why_selected",
    ]

    now        = pd.Timestamp.now()
    asof_date  = now.strftime("%Y-%m-%d")

    if df is None or df.empty:
        print("Ingen kandidater funnet. Skriver tom CSV.")
        df_out = pd.DataFrame(columns=expected_cols)
    else:
        df_out = df.copy()
        for c in expected_cols:
            if c not in df_out.columns:
                df_out[c] = np.nan
        asof_date = str(df_out["asof_date"].iloc[0]) if len(df_out) > 0 else asof_date

    latest_path  = "./top_candidates_latest.csv"
    history_path = f"./top_candidates_{asof_date}.csv"

    df_out.to_csv(latest_path, index=False)
    print(f"Latest fil skrevet: {latest_path}")

    df_out.to_csv(history_path, index=False)
    print(f"Historikk-fil skrevet: {history_path}")

    print("--- Pipeline fullført. ---")


if __name__ == "__main__":
    final_result_df = run_full_screener()
    export_results(final_result_df)
