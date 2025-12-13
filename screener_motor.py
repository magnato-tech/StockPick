import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
import os 
from typing import List, Dict, Any

# --- KONSTANTER ---
MARKET_INDEX_TICKER = 'SPY' 
TRAIN_RATIO = 0.70 
MIN_TEST_DAYS = 60 
MIN_DOLLAR_VOLUME = 500000 
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

# --- HJELPEFUNKSJONER ---

def get_master_ticker_list() -> pd.DataFrame:
    """Henter listen fra Wikipedia, med fallback til cachet fil."""
    print("-> Starter Henting av Masterliste.")
    
    # 1. Prøv Wikipedia først og cache resultatet
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp500_df = pd.read_html(url)[0]
        df = sp500_df[['Symbol', 'Security', 'GICS Sector']].rename(
            columns={'Symbol': 'ticker', 'Security': 'name', 'GICS Sector': 'sector'}
        )
        # Lagrer en kopi i repoet som fallback
        df.to_csv(MASTER_LIST_CACHE, index=False) 
        print("-> Liste hentet fra Wikipedia og cachet.")
        return df
    except Exception:
        print("-> Feil ved henting fra Wikipedia. Forsøker cachet fil.")
        
    # 2. Fallback til cachet fil
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
    if df.shape[0] < 50: 
        return None 
    
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
    """
    Simulerer en Trailing Stop Loss handel og returnerer en serie av kumulativ avkastning.
    """
    periode_data = df[df.index >= kjops_dato].copy()
    if periode_data.empty: return pd.Series(dtype=float)

    hoyeste_pris = kjops_pris
    posisjon_verdi = pd.Series(index=periode_data.index, dtype=float)
    exit_dato = None
    
    # Initierer dag 1
    posisjon_verdi.iloc[0] = kjops_pris
    
    for i, (dato, row) in enumerate(periode_data.iterrows()):
        if row['High'] > hoyeste_pris:
            hoyeste_pris = row['High']
            
        stop_niva = hoyeste_pris * (1 - stop_loss_pct)
        
        # Hvis Low treffer SL, avslutt handel
        if row['Low'] <= stop_niva:
            salgspris = stop_niva
            posisjon_verdi.iloc[i:] = salgspris
            exit_dato = dato
            break
        
        # Hvis handelen ikke avsluttes, oppdateres posisjonsverdien basert på sluttkursen
        posisjon_verdi.iloc[i] = row['Close']

    # Hvis handelen aldri avsluttet av SL
    if exit_dato is None:
        posisjon_verdi.iloc[i:] = periode_data.iloc[-1]['Close']
        
    # Konverter til avkastning i forhold til kjøpsprisen
    returns = (posisjon_verdi / kjops_pris) - 1
    return returns

def calculate_cagr(returns: pd.Series) -> float:
    """Beregner Compound Annual Growth Rate (CAGR) fra avkastningsserien."""
    if returns.empty: return 0.0
    
    total_returns = returns.iloc[-1] + 1.0 
    antall_dager = (returns.index[-1] - returns.index[0]).days
    
    if antall_dager <= 0: return 0.0
    
    years = antall_dager / 365.25
    return (total_returns ** (1 / years)) - 1.0

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Beregner den maksimale drawdown (tap) som andel."""
    if returns.empty: return 0.0
    
    # Verdi av posisjonen basert på kjøpspris (startverdi=1)
    equity_curve = returns + 1.0 
    
    peak = equity_curve.cummax()
    drawdown = (equity_curve / peak) - 1
    
    return abs(drawdown.min()) # Returneres som positiv andel (f.eks. 0.32)

def run_robust_vb_simulation(df_full: pd.DataFrame) -> Dict[str, Any]:
    """
    Kjører Pass 2: Train/Test split, SL optimalisering på Train, evaluering på Test.
    """
    if df_full.shape[0] < 200: 
        return None
    
    split_idx = int(df_full.shape[0] * TRAIN_RATIO)
    df_train = df_full.iloc[:split_idx].copy()
    df_test = df_full.iloc[split_idx:].copy()

    if df_test.shape[0] < MIN_TEST_DAYS:
        return None

    # Identifiser Kjøpspunkt i Train-settet (Laveste kurs i Train-perioden)
    min_row_train = df_train.loc[df_train['Low'].idxmin()]
    kjops_dato = min_row_train.name
    kjops_pris = min_row_train['Low']

    # --- 1. Optimalisering på TRAIN-settet (Grid Search) ---
    best_sl_train = 0.03 
    max_train_return = -1.0 
    
    sl_gitter = [sl / 100.0 for sl in range(3, 81, 1)] 
    
    for sl_test in sl_gitter:
        returns = simuler_handel_for_sim(df_train, kjops_dato, kjops_pris, sl_test)
        
        if not returns.empty and returns.iloc[-1] > max_train_return:
            max_train_return = returns.iloc[-1]
            best_sl_train = sl_test
            
    # --- 2. Evaluering på TEST-settet (Unngå Overtilpasning) ---
    
    # Kjøpsdato settes til den første dagen i Test-perioden.
    kjops_dato_test = df_test.index[0]
    kjops_pris_test = df_test.iloc[0]['Open']
    
    # Kjører simulering over Test-settet
    test_returns = simuler_handel_for_sim(df_test, kjops_dato_test, kjops_pris_test, best_sl_train)

    if test_returns.empty: return None

    # Beregn KPIer for V/B Score
    cagr_test = calculate_cagr(test_returns)
    max_dd_test = calculate_max_drawdown(test_returns)

    # --- 3. Robust V/B-ratio ---
    max_dd_test_floored = max(max_dd_test, 0.05) 
    
    if cagr_test <= 0:
        vb_ratio = 0.0
    else:
        # V/B Ratio: CAGR (andel) / Max Drawdown (andel)
        vb_ratio = cagr_test / max_dd_test_floored
    
    return {
        'vb_ratio': vb_ratio,
        'optimal_sl_train': best_sl_train * 100,
        'cagr_test_percent': cagr_test * 100,
        'max_drawdown_test': max_dd_test * 100,
    }


# --- HOVEDSCREENER ---

def run_full_screener():
    
    # 0. Pass 0: Regime Check (Tidlig Exit i Bear Market)
    print("\n--- 0. Pass 0: Regime Check ---")
    try:
        index_data = yf.download(MARKET_INDEX_TICKER, period='2y', interval='1d', progress=False)
        index_data['SMA200'] = index_data['Close'].rolling(window=200).mean()
        
        index_sma200 = index_data['SMA200'].iloc[-1]
        index_close = index_data['Close'].iloc[-1]
        
        if pd.isna(index_sma200) or index_close < index_sma200:
            print(f"MARKNEDSREGIME: Bearish/Ubestemt. ({MARKET_INDEX_TICKER} under SMA200).")
            print("Screening avsluttes tidlig.")
            return pd.DataFrame() 
        else:
            print(f"MARKNEDSREGIME: Bullish. ({MARKET_INDEX_TICKER} er over SMA200).")
            
    except Exception as e:
        print(f"Advarsel: Kunne ikke sjekke regime for {MARKET_INDEX_TICKER}. Fortsetter. Detaljer: {e}")
        
    
    # 1. Hent Masterliste
    ticker_list_df = get_master_ticker_list()
    if ticker_list_df.empty: return pd.DataFrame()
    
    print(f"\nStarter screening av {len(ticker_list_df)} aksjer.")
    
    # 2. Pass 1: Grovfiltrering (Likviditet/Momentum)
    print("\n--- 1. Pass 1: Grovfiltrering (Likviditet/Momentum) ---")
    shortlist_data = []
    
    for _, row in ticker_list_df.iterrows():
        ticker = row['ticker']
        time.sleep(SLEEP_TIME_SECONDS) # RATE LIMIT PAUSE
        
        try:
            # Bruk 1 år for å sikre SMA50 og volum. Legger til enkel retry.
            try:
                data = yf.download(ticker, period='1y', interval='1d', progress=False)
            except Exception:
                 time.sleep(SLEEP_TIME_SECONDS * 5)
                 data = yf.download(ticker, period='1y', interval='1d', progress=False)
                 
            metrics = calculate_technical_metrics(data, ticker)
            
            if metrics:
                # Filter 1: Likviditet (Min. $500,000 i gj.snitt daglig volum)
                if metrics['avg_dollar_vol_50'] < MIN_DOLLAR_VOLUME:
                    continue 

                # Filter 2: Momentum (Må være over SMA50, dvs. MomentumRatio > 0)
                if metrics['momentum_ratio'] <= 0:
                    continue

                shortlist_data.append({**row.to_dict(), **metrics})
            
        except Exception as e:
            # print(f"Feil ved henting/analyse av {ticker}: {e}")
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
    if final_df.empty: return pd.DataFrame()

    # 4. Pass 3: Normalisering og Vektet Score (Kommer i STEG 2.4)
    print("\n--- 3. Pass 3: Normalisering og Scoring ---")
    
    # DUMMY DATA FOR TESTING (Fylles ut i neste steg)
    if not final_df.empty:
        final_df['vb_percentile'] = np.random.rand(len(final_df)) * 100
        final_df['mom_percentile'] = final_df['momentum_ratio'].rank(pct=True) * 100
        final_df['vol_percentile'] = final_df['vol_ratio'].rank(pct=True) * 100
        final_df['total_score'] = (final_df['mom_percentile'] + final_df['vol_percentile']) / 2
        final_df['run_id'] = 'DUMMY_ID'

    return final_df.head(50) 

# --- EKSPORT (UTENFOR HOVEDFUNKSJONEN) ---

def export_results(df: pd.DataFrame):
    """
    Lagrer latest CSV og en datostemplet CSV.
    """
    if df.empty:
        print("Ingen kandidater funnet. Skriver ikke fil.")
        return

    # Setter siste handelsdag som asof_date
    asof_date = df['asof_date'].iloc[0]
    
    # Filnavn for Latest (overskrives)
    latest_filename = "top_candidates_latest.csv"
    df.to_csv(latest_filename, index=False)
    print(f"Latest fil skrevet: {latest_filename}")
    
    # Filnavn for Historikk (dato-stemplet)
    history_filename = f"top_candidates_{asof_date}.csv"
    df.to_csv(history_filename, index=False)
    print(f"Historikk-fil skrevet: {history_filename}")
        
    print("--- Pipeline Fullført. ---")
    

if __name__ == '__main__':
    final_result_df = run_full_screener()
    export_results(final_result_df)