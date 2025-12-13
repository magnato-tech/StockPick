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
MIN_DOLLAR_VOLUME = 500000 # Filter: Minst $500k i gj.snitt daglig volum
TIME_PERIOD_YEARS = 2 
SLEEP_TIME_SECONDS = 0.2
MIN_CANDIDATES_FOR_SECTOR_RANK = 20
MASTER_LIST_CACHE = 'sp500_master_cache.csv'

# --- VEKTING AV SCORE (Final Score) ---
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
    Returnerer None ved for kort / hull data.
    """
    # Må ha minst 50 dager for SMA50/AvgDollarVol50
    if df.shape[0] < 50: 
        # print(f"Advarsel: {ticker} har for lite data.")
        return None 
    
    # Prissjekk
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['MomentumRatio'] = df['Close'] / df['SMA50'] - 1
    
    # Volum/Likviditet
    df['DollarVolume'] = df['Close'] * df['Volume']
    df['AvgDollarVol5'] = df['DollarVolume'].rolling(window=5).mean()
    df['AvgDollarVol50'] = df['DollarVolume'].rolling(window=50).mean()
    df['VolRatio'] = df['AvgDollarVol5'] / df['AvgDollarVol50'] - 1
    
    # Siste rad (asof dato)
    last_row = df.iloc[-1]
    
    # NaN GUARD: Sjekker at metrikkene faktisk er beregnet (ikke NaN pga. for kort/hull data)
    # Dette er kritisk for å sikre tall i neste steg
    if pd.isna(last_row['SMA50']) or pd.isna(last_row['AvgDollarVol50']):
        # print(f"Advarsel: {ticker} SMA50/AvgDolVol50 er NaN.")
        return None

    metrics = {
        'ticker': ticker,
        'asof_date': df.index[-1].strftime('%Y-%m-%d'),
        'momentum_ratio': last_row['MomentumRatio'],
        'vol_ratio': last_row['VolRatio'],
        'avg_dollar_vol_50': last_row['AvgDollarVol50']
    }
    return metrics

def run_robust_vb_simulation(df_full: pd.DataFrame) -> Dict[str, Any]:
    """Kjører Pass 2: Train/Test split, SL optimalisering. Denne fylles ut i STEG 2.3."""
    return None

# --- HOVEDSCREENER ---

def run_full_screener():
    
    # 0. Pass 0: Regime Check (Tidlig Exit i Bear Market)
    print("\n--- 0. Pass 0: Regime Check ---")
    try:
        # Henter 2 år med data for å sikre at SMA200 er beregnet
        index_data = yf.download(MARKET_INDEX_TICKER, period='2y', interval='1d', progress=False)
        index_data['SMA200'] = index_data['Close'].rolling(window=200).mean()
        
        index_sma200 = index_data['SMA200'].iloc[-1]
        index_close = index_data['Close'].iloc[-1]
        
        # Sjekker at data er gyldig (ikke NaN) og at prisen er over snittet.
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

    # 3. Pass 2: Dyp Analyse (Kommer i STEG 2.3)
    print("\n--- 2. Pass 2: Dyp Analyse (V/B Simulering) ---")
    final_candidates = []
    
    for _, row in shortlist_df.iterrows():
        # DUMMY LØKKE, skal fylles ut i neste steg
        final_candidates.append(row.to_dict())
    
    final_df = pd.DataFrame(final_candidates)
    
    # 4. Pass 3: Normalisering og Vektet Score (Kommer i STEG 2.4)
    print("\n--- 3. Pass 3: Normalisering og Scoring ---")
    
    # DUMMY DATA FOR TESTING
    if not final_df.empty:
        final_df['vb_ratio'] = np.random.rand(len(final_df))
        final_df['mom_percentile'] = np.random.rand(len(final_df)) * 100
        final_df['vol_percentile'] = np.random.rand(len(final_df)) * 100
        final_df['total_score'] = (final_df['mom_percentile'] + final_df['vol_percentile']) / 2
        final_df['asof_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        final_df['run_id'] = 'DUMMY_ID'
        # Legg til dummy kolonner for outputkontrakten
        final_df['optimal_sl_train'] = 15.0
        final_df['cagr_test_percent'] = 20.0
        final_df['max_drawdown_test'] = 10.0
        final_df['why_selected'] = 'DUMMY VURDERING'


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