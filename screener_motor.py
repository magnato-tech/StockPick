import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import os 
from typing import List, Dict, Any, Tuple

# --- KONSTANTER ---
MARKET_INDEX_TICKER = 'SPY' 
TRAIN_RATIO = 0.70 
MIN_TEST_DAYS = 60 
MIN_DOLLAR_VOLUME = 500000 # Filter: Minst $500k i gj.snitt daglig volum
TIME_PERIOD_YEARS = 2 
SLEEP_TIME_SECONDS = 0.2
MIN_CANDIDATES_FOR_SECTOR_RANK = 20
MASTER_LIST_CACHE = 'sp500_master_cache.csv'

# --- VEKTING AV SCORE ---
WEIGHTS = {
    'vb_percentile': 0.50,
    'mom_percentile': 0.30,
    'vol_percentile': 0.20
}

# --- HJELPEFUNKSJONER (Data Henting & Validering) ---

def get_master_ticker_list() -> pd.DataFrame:
    """Henter listen fra Wikipedia, med fallback til cachet fil."""
    print("-> Starter Henting av Masterliste.")
    
    # Prøv Wikipedia først og cache resultatet
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp500_df = pd.read_html(url)[0]
        df = sp500_df[['Symbol', 'Security', 'GICS Sector']].rename(
            columns={'Symbol': 'ticker', 'Security': 'name', 'GICS Sector': 'sector'}
        )
        df.to_csv(MASTER_LIST_CACHE, index=False) 
        print("-> Liste hentet fra Wikipedia og cachet.")
        return df
    except Exception:
        print("-> Feil ved henting fra Wikipedia. Forsøker cachet fil.")
        
    # Fallback til cachet fil
    if os.path.exists(MASTER_LIST_CACHE):
        df = pd.read_csv(MASTER_LIST_CACHE)
        print("-> Liste hentet fra cache.")
        return df
    else:
        print("-> KRITISK FEIL: Ingen masterliste funnet.")
        return pd.DataFrame(columns=['ticker', 'name', 'sector'])

def calculate_technical_metrics(df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """
    Kjører Pass 1: Beregner tekniske metrikker og legger inn NaN-guards.
    """
    if df.shape[0] < 50: return None 
    
    # Prissjekk
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['MomentumRatio'] = df['Close'] / df['SMA50'] - 1
    
    # Volum/Likviditet
    df['DollarVolume'] = df['Close'] * df['Volume']
    df['AvgDollarVol5'] = df['DollarVolume'].rolling(window=5).mean()
    df['AvgDollarVol50'] = df['DollarVolume'].rolling(window=50).mean()
    df['VolRatio'] = df['AvgDollarVol5'] / df['AvgDollarVol50'] - 1
    
    last_row = df.iloc[-1]
    
    # NaN GUARD: Sjekker at metrikkene faktisk er beregnet
    if pd.isna(last_row['SMA50']) or pd.isna(last_row['AvgDollarVol50']):
        return None

    metrics = {
        'ticker': ticker,
        'asof_date': df.index[-1].strftime('%Y-%m-%d'),
        'momentum_ratio': last_row['MomentumRatio'],
        'vol_ratio': last_row['VolRatio'],
        'avg_dollar_vol_50': last_row['AvgDollarVol50']
    }
    return metrics

# --- FUNKSJONER FOR PASS 2 (V/B SIMULERING) ---

def simuler_handel_for_sim(df, kjops_dato, kjops_pris, stop_loss_pct):
    """Simulerer Trailing Stop Loss handel og returnerer en serie av kumulativ avkastning."""
    periode_data = df[df.index >= kjops_dato].copy()
    if periode_data.empty: return pd.Series(dtype=float)

    hoyeste_pris = kjops_pris
    posisjon_verdi = pd.Series(index=periode_data.index, dtype=float)
    exit_dato = None
    
    posisjon_verdi.iloc[0] = kjops_pris
    
    for i, (dato, row) in enumerate(periode_data.iterrows()):
        if row['High'] > hoyeste_pris:
            hoyeste_pris = row['High']
            
        stop_niva = hoyeste_pris * (1 - stop_loss_pct)
        
        if row['Low'] <= stop_niva:
            salgspris = stop_niva
            posisjon_verdi.iloc[i:] = salgspris
            exit_dato = dato
            break
        
        posisjon_verdi.iloc[i] = row['Close']

    if exit_dato is None:
        posisjon_verdi.iloc[i:] = periode_data.iloc[-1]['Close']
        
    returns = (posisjon_verdi / kjops_pris) - 1
    return returns

def calculate_cagr(returns: pd.Series) -> float:
    """Beregner Compound Annual Growth Rate (CAGR)."""
    if returns.empty: return 0.0
    
    total_returns = returns.iloc[-1] + 1.0 
    antall_dager = (returns.index[-1] - returns.index[0]).days
    
    if antall_dager <= 0: return 0.0
    
    years = antall_dager / 365.25
    return (total_returns ** (1 / years)) - 1.0

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Beregner den maksimale drawdown (tap) som andel."""
    if returns.empty: return 0.0
    
    equity_curve = returns + 1.0 
    peak = equity_curve.cummax()
    drawdown = (equity_curve / peak) - 1
    
    return abs(drawdown.min()) # Returneres som positiv andel

# --- FUNKSJONER FOR ROBUST MULTI-START OPTIMALISERING ---

def pick_start_dates(df: pd.DataFrame, min_lookback: int = 60, step: int = 10) -> List[pd.Timestamp]:
    """Velger mulige startdatoer med faste intervaller (hver 10. dag)."""
    if len(df) <= min_lookback + 5:
        return []
        
    # Velger hver 'step'-te handelsdag etter min_lookback
    idx = df.index[min_lookback::step]
    return list(idx)

def grid_search_sl_multi_start(
    df: pd.DataFrame,
    sl_range: range,
    start_dates: List[pd.Timestamp],
    use_percentile: int = 50, # 50 = median
) -> Tuple[float, pd.DataFrame]:
    """
    Kjører stop-loss grid search på flere startdatoer for å finne optimal SL.
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
            
            # Kjøper på Close samme dag for enkelhet og konsistens
            buy_price = float(df.loc[start, "Close"])
            
            # Får hele avkastningsserien, bruker sluttverdien (iloc[-1]) for scoring
            returns_series = simuler_handel_for_sim(df, start, buy_price, sl_dec)
            
            if not returns_series.empty:
                 per_start_returns.append(returns_series.iloc[-1])

        if not per_start_returns:
            continue

        # Beregner Median Score (50. percentil)
        score = float(np.percentile(per_start_returns, use_percentile))
        
        results.append({
            "stop_loss_pct": sl,
            "score": score,
        })

    res_df = pd.DataFrame(results).sort_values("score", ascending=False)
    
    if res_df.empty:
        return None, res_df

    # Returnerer optimal SL som andel (for å brukes i Pass 2)
    best_sl_pct = res_df.iloc[0]["stop_loss_pct"] / 100.0
    return best_sl_pct, res_df


def run_robust_vb_simulation(df_full: pd.DataFrame) -> Dict[str, Any]:
    """
    Kjører Pass 2: Train/Test split, SL optimalisering (Multi-Start) på Train, evaluering på Test.
    """
    if df_full.shape[0] < 200: 
        return None
    
    split_idx = int(df_full.shape[0] * TRAIN_RATIO)
    df_train = df_full.iloc[:split_idx].copy()
    df_test = df_full.iloc[split_idx:].copy()

    if df_test.shape[0] < MIN_TEST_DAYS:
        return None

    # --- 1. Optimalisering på TRAIN-settet (Multi-Start Grid Search) ---
    train_start_dates = pick_start_dates(df_train, min_lookback=60, step=10)
    sl_range_pct = range(3, 81, 1)

    # Finn optimal SL (returneres som andel)
    best_sl_train, _ = grid_search_sl_multi_start(
        df_train,
        sl_range=sl_range_pct,
        start_dates=train_start_dates,
        use_percentile=50
    )
    
    if best_sl_train is None: 
        return None

    # --- 2. Evaluering på TEST-settet (Enkelt Startpunkt) ---
    
    # Kjøpsdato settes til den første dagen i Test-perioden.
    kjops_dato_test = df_test.index[0]
    kjops_pris_test = df_test.iloc[0]['Open']
    
    # Kjører simulering over Test-settet med den OPTIMALE SL fra Train-settet
    test_returns = simuler_handel_for_sim(df_test, kjops_dato_test, kjops_pris_test, best_sl_train)

    if test_returns.empty: return None

    # Beregn KPIer for V/B Score
    cagr_test = calculate_cagr(test_returns)
    max_dd_test = calculate_max_drawdown(test_returns)

    # --- 3. Robust V/B-ratio ---
    max_dd_test_floored = max(max_dd_test, 0.05) # Gulv på 5% Drawdown
    
    if cagr_test <= 0:
        vb_ratio = 0.0
    else:
        vb_ratio = cagr_test / max_dd_test_floored
    
    return {
        'vb_ratio': vb_ratio,
        'optimal_sl_train': best_sl_train * 100, # Konverteres til prosent
        'cagr_test_percent': cagr_test * 100,
        'max_drawdown_test': max_dd_test * 100,
    }


# --- HOVEDSCREENER ---

def run_full_screener():
    
    # 0. Pass 0: Regime Check
    print("\n--- 0. Pass 0: Regime Check ---")
    try:
        index_data = yf.download(MARKET_INDEX_TICKER, period='2y', interval='1d', progress=False)
        index_data['SMA200'] = index_data['Close'].rolling(window=200).mean()
        index_sma200 = index_data['SMA200'].iloc[-1]
        index_close = index_data['Close'].iloc[-1]
        
        if pd.isna(index_sma200) or index_close < index_sma200:
            print(f"MARKNEDSREGIME: Bearish/Ubestemt. Screening avsluttes tidlig.")
            return pd.DataFrame() 
        else:
            print(f"MARKNEDSREGIME: Bullish.")
            
    except Exception as e:
        print(f"Advarsel: Kunne ikke sjekke regime. Fortsetter. Detaljer: {e}")
        
    
    # 1. Hent Masterliste
    ticker_list_df = get_master_ticker_list()
    if ticker_list_df.empty: return pd.DataFrame()
    
    print(f"\nStarter screening av {len(ticker_list_df)} aksjer.")
    
    # 2. Pass 1: Grovfiltrering
    print("\n--- 1. Pass 1: Grovfiltrering (Likviditet/Momentum) ---")
    shortlist_data = []
    
    for _, row in ticker_list_df.iterrows():
        ticker = row['ticker']
        time.sleep(SLEEP_TIME_SECONDS) # RATE LIMIT PAUSE
        
        try:
            # Bruk 1 år for å sikre SMA50 og volum. Legger til enkel retry.
            data = yf.download(ticker, period='1y', interval='1d', progress=False)
            metrics = calculate_technical_metrics(data, ticker)
            
            if metrics:
                # Filter 1: Likviditet
                if metrics['avg_dollar_vol_50'] < MIN_DOLLAR_VOLUME:
                    continue 
                # Filter 2: Momentum
                if metrics['momentum_ratio'] <= 0:
                    continue

                shortlist_data.append({**row.to_dict(), **metrics})
            
        except Exception:
            continue

    shortlist_df = pd.DataFrame(shortlist_data)
    if shortlist_df.empty:
        print("Ingen aksjer kvalifisert etter Pass 1.")
        return pd.DataFrame()

    print(f"Pass 1 fullført. {len(shortlist_df)} aksjer kvalifisert for Pass 2.")

    # 3. Pass 2: Dyp Analyse (V/B Simulering)
    print("\n--- 2. Pass 2: Dyp Analyse (V/B Simulering) ---")
    final_candidates = []
    
    for _, row in shortlist_df.iterrows():
        ticker = row['ticker']
        time.sleep(SLEEP_TIME_SECONDS) # RATE LIMIT PAUSE

        try:
            data = yf.download(ticker, period=f'{TIME_PERIOD_YEARS}y', interval='1d', progress=False)
            vb_results = run_robust_vb_simulation(data)
            
            if vb_results:
                final_candidates.append({**row.to_dict(), **vb_results})
                
        except Exception:
            continue
    
    final_df = pd.DataFrame(final_candidates)
    if final_df.empty: 
        print("Ingen aksjer kvalifisert etter Pass 2.")
        return pd.DataFrame()

    # 4. Pass 3: Normalisering og Vektet Score
    print("\n--- 3. Pass 3: Normalisering og Scoring ---")
    
    def robust_normalize(df: pd.DataFrame, col: str) -> np.ndarray:
        """Robust percentil-ranking med fallback til global rank for små sektorer."""
        sector_ranks = df.groupby('sector')[col].transform(lambda x: x.rank(pct=True) * 100)
        sector_counts = df.groupby('sector')[col].transform('count')
        global_rank = df[col].rank(pct=True) * 100
        
        return np.where(sector_counts >= MIN_CANDIDATES_FOR_SECTOR_RANK, sector_ranks, global_rank)

    # V/B Score
    final_df['vb_percentile'] = robust_normalize(final_df, 'vb_ratio')
    
    # Momentum Score
    final_df['mom_percentile'] = robust_normalize(final_df, 'momentum_ratio')
    
    # Volum Score (Winsorize først)
    q_low = final_df['vol_ratio'].quantile(0.01)
    q_high = final_df['vol_ratio'].quantile(0.99)
    final_df['vol_ratio_capped'] = final_df['vol_ratio'].clip(lower=q_low, upper=q_high)
    final_df['vol_percentile'] = robust_normalize(final_df, 'vol_ratio_capped')
    
    # Beregn Total Score
    final_df['total_score'] = (
        final_df['vb_percentile'] * WEIGHTS['vb_percentile'] +
        final_df['mom_percentile'] * WEIGHTS['mom_percentile'] +
        final_df['vol_percentile'] * WEIGHTS['vol_percentile']
    )

    # Klargjøring av Output
    final_df = final_df.sort_values(by='total_score', ascending=False).head(50).reset_index(drop=True)
    
    final_df['run_id'] = pd.Timestamp.now().strftime('%Y%m%d%H%M%S') 
    
    final_df['why_selected'] = final_df.apply(
        lambda x: (
            f"V/B Score: {x['vb_percentile']:.1f} / Mom: {x['momentum_ratio']*100:.1f}% "
            f"/ Vol: {x['vol_ratio']*100:.1f}% / SL: {x['optimal_sl_train']:.1f}%"
        ), 
        axis=1
    )
    
    # Velg de kolonnene som matcher den avtalte kontrakten
    output_cols = [
        'asof_date', 'run_id', 'ticker', 'name', 'sector', 'total_score', 
        'vb_percentile', 'mom_percentile', 'vol_percentile', 
        'optimal_sl_train', 'cagr_test_percent', 'max_drawdown_test', 
        'avg_dollar_vol_50', 'why_selected'
    ]
    
    output_cols = [col for col in output_cols if col in final_df.columns]
    
    return final_df[output_cols]

# --- EKSPORT (OPPDATERT FOR ROBUSTHET) ---

def export_results(df: pd.DataFrame):
    """
    Lagrer latest CSV og en datostemplet CSV til disk for GitHub Actions.
    Viktig: Skriver ALLTID latest-fila (også når df er tom), slik at Streamlit aldri får 404.
    """
    expected_cols = [
        'asof_date', 'run_id', 'ticker', 'name', 'sector', 'total_score',
        'vb_percentile', 'mom_percentile', 'vol_percentile',
        'optimal_sl_train', 'cagr_test_percent', 'max_drawdown_test',
        'why_selected'
    ]

    now = pd.Timestamp.now()
    asof_date = now.strftime('%Y-%m-%d')

    if df is None or df.empty:
        print("Ingen kandidater funnet. Skriver tom CSV (for å unngå 404 i Streamlit).")
        # Oppretter en tom DataFrame med riktige kolonner (kontrakten)
        df_out = pd.DataFrame(columns=expected_cols)
        # Legger på kjøringsinformasjon (tomme kolonner)
        df_out['asof_date'] = pd.Series(dtype=str)
        df_out['run_id'] = pd.Series(dtype=str)
    else:
        df_out = df.copy()
        # Sikrer at forventede kolonner finnes (Streamlit forventer disse)
        for c in expected_cols:
            if c not in df_out.columns:
                df_out[c] = np.nan

        # asof_date brukes til historikkfil
        asof_date = str(df_out['asof_date'].iloc[0]) if 'asof_date' in df_out.columns and len(df_out) > 0 else asof_date

    # Filbanefiks: Bruk './' for å tvinge lagring til rotkatalogen
    latest_filename = "./top_candidates_latest.csv"
    df_out.to_csv(latest_filename, index=False)
    print(f"Latest fil skrevet: {latest_filename}")

    history_filename = f"./top_candidates_{asof_date}.csv"
    df_out.to_csv(history_filename, index=False)
    print(f"Historikk-fil skrevet: {history_filename}")

    print("--- Pipeline Fullført. ---")


if __name__ == '__main__':
    final_result_df = run_full_screener()
    export_results(final_result_df)
