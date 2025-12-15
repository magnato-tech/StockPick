import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
from typing import List, Dict, Any, Tuple

# ============================================================
# KONSTANTER / PARAMETRE
# ============================================================

MARKET_INDEX_TICKER = "SPY"

TRAIN_RATIO = 0.70
MIN_TEST_DAYS = 60
TIME_PERIOD_YEARS = 2
SLEEP_TIME_SECONDS = 0.2

MIN_CANDIDATES_FOR_SECTOR_RANK = 20
MASTER_LIST_CACHE = "sp500_master_cache.csv"

# --- GATE: eneste absolutte krav ---
SMA50_CROSS_LOOKBACK_DAYS = 10  # ca. 2 uker (handelsdager)

# --- BONUS: langsiktig trendkvalitet ---
SMA150_BONUS_POINTS = 10.0

# --- VEKTING AV SCORE ---
WEIGHTS = {
    "vb_percentile": 0.50,
    "mom_percentile": 0.30,
    "vol_percentile": 0.20,
}

# ============================================================
# HJELPEFUNKSJONER (Masterliste)
# ============================================================

def get_master_ticker_list() -> pd.DataFrame:
    """Henter S&P 500 fra Wikipedia, med fallback til cachet fil."""
    print("-> Starter henting av masterliste (S&P 500).")

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp500_df = pd.read_html(url)[0]
        df = sp500_df[["Symbol", "Security", "GICS Sector"]].rename(
            columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"}
        )
        df.to_csv(MASTER_LIST_CACHE, index=False)
        print("-> Liste hentet fra Wikipedia og cachet.")
        return df
    except Exception as e:
        print(f"-> Feil ved henting fra Wikipedia. Forsøker cachet fil. Detaljer: {e}")

    if os.path.exists(MASTER_LIST_CACHE):
        df = pd.read_csv(MASTER_LIST_CACHE)
        print("-> Liste hentet fra cache.")
        return df

    print("-> KRITISK FEIL: Ingen masterliste funnet.")
    return pd.DataFrame(columns=["ticker", "name", "sector"])


# ============================================================
# PASS 1: Teknisk metrikk (inkl gate + bonus-flagg)
# ============================================================

def _days_since_last_true(bool_series: pd.Series) -> Any:
    """Returnerer antall dager siden siste True i en bool-serie (0=senest i dag)."""
    if bool_series is None or bool_series.empty:
        return None
    if not bool_series.any():
        return None
    # Finn siste indeks der True
    last_true_pos = np.where(bool_series.values)[0][-1]
    return int(len(bool_series) - 1 - last_true_pos)


def calculate_technical_metrics(df: pd.DataFrame, ticker: str) -> Dict[str, Any] | None:
    """
    Beregner tekniske metrikker og gates/bonus-signaler.
    Gate:
      - kryss opp over SMA50 innen siste N handelsdager
      - og fortsatt over SMA50 i dag
    Bonus:
      - kryss opp over SMA150 innen siste N handelsdager + fortsatt over i dag
    """
    if df is None or df.empty:
        return None

    # Vi trenger nok data til SMA150 + litt buffer
    if df.shape[0] < 160:
        return None

    # Sørg for at vi har standardkolonner
    required_cols = {"Close", "High", "Low", "Open", "Volume"}
    if not required_cols.issubset(set(df.columns)):
        return None

    # Glidende snitt
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA150"] = df["Close"].rolling(window=150).mean()

    # Momentum (fortsatt brukt til ranking)
    df["MomentumRatio"] = df["Close"] / df["SMA50"] - 1

    # Volum/Likviditet (fortsatt brukt til ranking)
    df["DollarVolume"] = df["Close"] * df["Volume"]
    df["AvgDollarVol5"] = df["DollarVolume"].rolling(window=5).mean()
    df["AvgDollarVol50"] = df["DollarVolume"].rolling(window=50).mean()
    df["VolRatio"] = df["AvgDollarVol5"] / df["AvgDollarVol50"] - 1

    # NaN-guard
    last_row = df.iloc[-1]
    if pd.isna(last_row["SMA50"]) or pd.isna(last_row["AvgDollarVol50"]) or pd.isna(last_row["SMA150"]):
        return None

    # --- GATE: SMA50 cross-up siste N dager + fortsatt over ---
    cross50_series = (df["Close"] > df["SMA50"]) & (df["Close"].shift(1) <= df["SMA50"].shift(1))
    cross50_last_n = bool(cross50_series.tail(SMA50_CROSS_LOOKBACK_DAYS).any())
    still_above_50 = bool(df["Close"].iloc[-1] > df["SMA50"].iloc[-1])
    days_since_cross50 = _days_since_last_true(cross50_series)

    # --- BONUS: SMA150 cross-up siste N dager + fortsatt over ---
    cross150_series = (df["Close"] > df["SMA150"]) & (df["Close"].shift(1) <= df["SMA150"].shift(1))
    cross150_recent = bool(cross150_series.tail(SMA50_CROSS_LOOKBACK_DAYS).any())
    still_above_150 = bool(df["Close"].iloc[-1] > df["SMA150"].iloc[-1])
    days_since_cross150 = _days_since_last_true(cross150_series)

    metrics = {
        "ticker": ticker,
        "asof_date": df.index[-1].strftime("%Y-%m-%d"),

        # Ranking-metrikker
        "momentum_ratio": float(last_row["MomentumRatio"]),
        "vol_ratio": float(last_row["VolRatio"]),
        "avg_dollar_vol_50": float(last_row["AvgDollarVol50"]),

        # Gate / signal
        "cross50_last_n": bool(cross50_last_n),
        "still_above_50": bool(still_above_50),
        "days_since_cross50": days_since_cross50,

        # Bonus-signal
        "cross150_recent": bool(cross150_recent),
        "still_above_150": bool(still_above_150),
        "days_since_cross150": days_since_cross150,
    }

    return metrics


# ============================================================
# PASS 2: V/B (robust simulering) – uendret logikk
# ============================================================

def simuler_handel_for_sim(df, kjops_dato, kjops_pris, stop_loss_pct):
    """Simulerer Trailing Stop Loss handel og returnerer en serie av kumulativ avkastning."""
    periode_data = df[df.index >= kjops_dato].copy()
    if periode_data.empty:
        return pd.Series(dtype=float)

    hoyeste_pris = kjops_pris
    posisjon_verdi = pd.Series(index=periode_data.index, dtype=float)
    exit_dato = None

    posisjon_verdi.iloc[0] = kjops_pris

    for i, (dato, row) in enumerate(periode_data.iterrows()):
        if row["High"] > hoyeste_pris:
            hoyeste_pris = row["High"]

        stop_niva = hoyeste_pris * (1 - stop_loss_pct)

        if row["Low"] <= stop_niva:
            salgspris = stop_niva
            posisjon_verdi.iloc[i:] = salgspris
            exit_dato = dato
            break

        posisjon_verdi.iloc[i] = row["Close"]

    if exit_dato is None:
        posisjon_verdi.iloc[i:] = periode_data.iloc[-1]["Close"]

    returns = (posisjon_verdi / kjops_pris) - 1
    return returns


def calculate_cagr(returns: pd.Series) -> float:
    """Beregner Compound Annual Growth Rate (CAGR)."""
    if returns.empty:
        return 0.0

    total_returns = returns.iloc[-1] + 1.0
    antall_dager = (returns.index[-1] - returns.index[0]).days
    if antall_dager <= 0:
        return 0.0

    years = antall_dager / 365.25
    return (total_returns ** (1 / years)) - 1.0


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Beregner den maksimale drawdown (tap) som andel."""
    if returns.empty:
        return 0.0

    equity_curve = returns + 1.0
    peak = equity_curve.cummax()
    drawdown = (equity_curve / peak) - 1
    return abs(drawdown.min())


def pick_start_dates(df: pd.DataFrame, min_lookback: int = 60, step: int = 10) -> List[pd.Timestamp]:
    """Velger mulige startdatoer med faste intervaller."""
    if len(df) <= min_lookback + 5:
        return []
    idx = df.index[min_lookback::step]
    return list(idx)


def grid_search_sl_multi_start(
    df: pd.DataFrame,
    sl_range: range,
    start_dates: List[pd.Timestamp],
    use_percentile: int = 50,
) -> Tuple[float | None, pd.DataFrame]:
    """
    Stop-loss grid search på flere startdatoer for å finne optimal SL (median).
    Returnerer: best_sl (andel), resultat-DataFrame
    """
    if not start_dates:
        return None, pd.DataFrame()

    results = []
    for sl in sl_range:
        sl_dec = sl / 100.0
        per_start_returns = []

        for start in start_dates:
            if start not in df.index:
                continue

            buy_price = float(df.loc[start, "Close"])
            returns_series = simuler_handel_for_sim(df, start, buy_price, sl_dec)
            if not returns_series.empty:
                per_start_returns.append(returns_series.iloc[-1])

        if not per_start_returns:
            continue

        score = float(np.percentile(per_start_returns, use_percentile))
        results.append({"stop_loss_pct": sl, "score": score})

    res_df = pd.DataFrame(results).sort_values("score", ascending=False)
    if res_df.empty:
        return None, res_df

    best_sl_pct = res_df.iloc[0]["stop_loss_pct"] / 100.0
    return best_sl_pct, res_df


def run_robust_vb_simulation(df_full: pd.DataFrame) -> Dict[str, Any] | None:
    """Train/Test split, SL optimalisering på Train, evaluering på Test."""
    if df_full is None or df_full.empty:
        return None

    if df_full.shape[0] < 200:
        return None

    split_idx = int(df_full.shape[0] * TRAIN_RATIO)
    df_train = df_full.iloc[:split_idx].copy()
    df_test = df_full.iloc[split_idx:].copy()

    if df_test.shape[0] < MIN_TEST_DAYS:
        return None

    train_start_dates = pick_start_dates(df_train, min_lookback=60, step=10)
    sl_range_pct = range(3, 81, 1)

    best_sl_train, _ = grid_search_sl_multi_start(
        df_train,
        sl_range=sl_range_pct,
        start_dates=train_start_dates,
        use_percentile=50,
    )
    if best_sl_train is None:
        return None

    # Kjøper første dag i test
    kjops_dato_test = df_test.index[0]
    kjops_pris_test = float(df_test.iloc[0]["Open"])

    test_returns = simuler_handel_for_sim(df_test, kjops_dato_test, kjops_pris_test, best_sl_train)
    if test_returns.empty:
        return None

    cagr_test = calculate_cagr(test_returns)
    max_dd_test = calculate_max_drawdown(test_returns)

    # Robust V/B ratio
    max_dd_test_floored = max(max_dd_test, 0.05)  # gulv 5%
    if cagr_test <= 0:
        vb_ratio = 0.0
    else:
        vb_ratio = cagr_test / max_dd_test_floored

    return {
        "vb_ratio": vb_ratio,
        "optimal_sl_train": best_sl_train * 100,
        "cagr_test_percent": cagr_test * 100,
        "max_drawdown_test": max_dd_test * 100,
    }


# ============================================================
# HOVEDSCREENER
# ============================================================

def run_full_screener():
    # 0) Pass 0: Regime Check (IKKE hard stop – kun info)
    print("\n--- 0. Pass 0: Regime Check ---")
    try:
        index_data = yf.download(MARKET_INDEX_TICKER, period="2y", interval="1d", progress=False)
        index_data["SMA200"] = index_data["Close"].rolling(window=200).mean()
        index_sma200 = index_data["SMA200"].iloc[-1]
        index_close = index_data["Close"].iloc[-1]

        if pd.isna(index_sma200) or index_close < index_sma200:
            print("MARKNEDSREGIME: Bearish/Ubestemt. Fortsetter screening likevel.")
        else:
            print("MARKNEDSREGIME: Bullish.")
    except Exception as e:
        print(f"Advarsel: Kunne ikke sjekke regime. Fortsetter. Detaljer: {e}")

    # 1) Masterliste
    ticker_list_df = get_master_ticker_list()
    if ticker_list_df.empty:
        return pd.DataFrame()

    print(f"\nStarter screening av {len(ticker_list_df)} aksjer.")

    # 2) Pass 1: Gate på SMA50-cross
    print("\n--- 1. Pass 1: Gate (SMA50-cross) + innsamling av metrikker ---")
    shortlist_data = []

    for _, row in ticker_list_df.iterrows():
        ticker = row["ticker"]
        time.sleep(SLEEP_TIME_SECONDS)

        try:
            data = yf.download(ticker, period="1y", interval="1d", progress=False)
            metrics = calculate_technical_metrics(data, ticker)

            if metrics is None:
                continue

            # ENESTE ABSOLUTTE KRAV:
            if not metrics.get("cross50_last_n", False):
                continue
            if not metrics.get("still_above_50", False):
                continue

            shortlist_data.append({**row.to_dict(), **metrics})

        except Exception:
            continue

    shortlist_df = pd.DataFrame(shortlist_data)
    if shortlist_df.empty:
        print("Ingen aksjer kvalifisert etter Pass 1 (SMA50-cross gate).")
        return pd.DataFrame()

    print(f"Pass 1 fullført. {len(shortlist_df)} aksjer kvalifisert for Pass 2.")

    # 3) Pass 2: V/B simulering
    print("\n--- 2. Pass 2: Dyp Analyse (V/B Simulering) ---")
    final_candidates = []

    for _, row in shortlist_df.iterrows():
        ticker = row["ticker"]
        time.sleep(SLEEP_TIME_SECONDS)

        try:
            data = yf.download(ticker, period=f"{TIME_PERIOD_YEARS}y", interval="1d", progress=False)
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
        """Percentil-ranking med fallback til global rank for små sektorer."""
        sector_ranks = df.groupby("sector")[col].transform(lambda x: x.rank(pct=True) * 100)
        sector_counts = df.groupby("sector")[col].transform("count")
        global_rank = df[col].rank(pct=True) * 100
        return np.where(sector_counts >= MIN_CANDIDATES_FOR_SECTOR_RANK, sector_ranks, global_rank)

    # Percentiler
    final_df["vb_percentile"] = robust_normalize(final_df, "vb_ratio")
    final_df["mom_percentile"] = robust_normalize(final_df, "momentum_ratio")

    # Volumscore: winsorize vol_ratio før percentil
    q_low = final_df["vol_ratio"].quantile(0.01)
    q_high = final_df["vol_ratio"].quantile(0.99)
    final_df["vol_ratio_capped"] = final_df["vol_ratio"].clip(lower=q_low, upper=q_high)
    final_df["vol_percentile"] = robust_normalize(final_df, "vol_ratio_capped")

    # Base total score
    final_df["total_score"] = (
        final_df["vb_percentile"] * WEIGHTS["vb_percentile"]
        + final_df["mom_percentile"] * WEIGHTS["mom_percentile"]
        + final_df["vol_percentile"] * WEIGHTS["vol_percentile"]
    )

    # Bonus: SMA150 trend-kvalitet
    final_df["sma150_bonus"] = 0.0
    mask150 = (final_df.get("cross150_recent", False) == True) & (final_df.get("still_above_150", False) == True)
    final_df.loc[mask150, "sma150_bonus"] = SMA150_BONUS_POINTS
    final_df["total_score"] = final_df["total_score"] + final_df["sma150_bonus"]

    # Klargjør output
    final_df = final_df.sort_values(by="total_score", ascending=False).head(50).reset_index(drop=True)

    final_df["run_id"] = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")

    def _fmt(x, default="n/a"):
        try:
            if pd.isna(x):
                return default
            return x
        except Exception:
            return default

    final_df["why_selected"] = final_df.apply(
        lambda x: (
            f"SMA50-cross: {int(_fmt(x.get('days_since_cross50'), 999))}d siden"
            f" | Over SMA50: {bool(_fmt(x.get('still_above_50'), False))}"
            f" | SMA150+: {bool(_fmt(x.get('cross150_recent'), False) and _fmt(x.get('still_above_150'), False))}"
            f" | V/B pct: {x['vb_percentile']:.1f}"
            f" | Mom: {x.get('momentum_ratio', 0.0)*100:.1f}%"
            f" | Vol: {x.get('vol_ratio', 0.0)*100:.1f}%"
            f" | SL: {x.get('optimal_sl_train', np.nan):.1f}%"
        ),
        axis=1,
    )

    output_cols = [
        "asof_date",
        "run_id",
        "ticker",
        "name",
        "sector",
        "total_score",
        "vb_percentile",
        "mom_percentile",
        "vol_percentile",
        "optimal_sl_train",
        "cagr_test_percent",
        "max_drawdown_test",
        "avg_dollar_vol_50",
        "why_selected",
    ]

    # Sikre at alle kolonnene finnes
    for c in output_cols:
        if c not in final_df.columns:
            final_df[c] = np.nan

    return final_df[output_cols]


# ============================================================
# EKSPORT (ALLTID skriv latest + historikk)
# ============================================================

def export_results(df: pd.DataFrame):
    """
    Lagrer latest CSV og en datostemplet CSV til disk for GitHub Actions.
    Skriver ALLTID latest (også når df er tom), så Streamlit aldri får 404.
    """
    expected_cols = [
        "asof_date",
        "run_id",
        "ticker",
        "name",
        "sector",
        "total_score",
        "vb_percentile",
        "mom_percentile",
        "vol_percentile",
        "optimal_sl_train",
        "cagr_test_percent",
        "max_drawdown_test",
        "avg_dollar_vol_50",
        "why_selected",
    ]

    now = pd.Timestamp.now()
    asof_date = now.strftime("%Y-%m-%d")

    if df is None or df.empty:
        print("Ingen kandidater funnet. Skriver tom CSV (for å unngå 404 i Streamlit).")
        df_out = pd.DataFrame(columns=expected_cols)
        df_out["asof_date"] = pd.Series(dtype=str)
        df_out["run_id"] = pd.Series(dtype=str)
    else:
        df_out = df.copy()
        for c in expected_cols:
            if c not in df_out.columns:
                df_out[c] = np.nan

        asof_date = str(df_out["asof_date"].iloc[0]) if "asof_date" in df_out.columns and len(df_out) > 0 else asof_date

    latest_filename = "./top_candidates_latest.csv"
    df_out.to_csv(latest_filename, index=False)
    print(f"Latest fil skrevet: {latest_filename}")

    history_filename = f"./top_candidates_{asof_date}.csv"
    df_out.to_csv(history_filename, index=False)
    print(f"Historikk-fil skrevet: {history_filename}")

    print("--- Pipeline fullført. ---")


if __name__ == "__main__":
    final_result_df = run_full_screener()
    export_results(final_result_df)
