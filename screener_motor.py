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

# --- KJERNEFUNKSJONER ---

def calculate_technical_metrics(df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
    """Beregn tekniske metrikker for Pass 1. Denne fylles ut i STEG 2.2."""
    # Må ha minst 50 dager for SMA50
    if df.shape[0] < 50: return None 
    
    # ... (LOGIKK KOMMER I NESTE STEG) ...
    
    # Returnerer None hvis det mangler data
    return None

def run_robust_vb_simulation(df_full: pd.DataFrame) -> Dict[str, Any]:
    """Kjører Pass 2: Train/Test split, SL optimalisering. Denne fylles ut i STEG 2.3."""
    return None

# --- HOVEDSCREENER ---

def run_full_screener():
    
    # 0. Pass 0: Regime Check (Tidlig Exit i Bear Market)
    print("\n--- 0. Pass 0: Regime Check ---")
    try:
        # Henter 2 år med data for å sikre at SMA200 er beregnet, selv med hull.
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
    
    # 2. Pass 1: Grovfiltrering (Kommer i STEG 2.2)
    print("\n--- 1. Pass 1: Grovfiltrering (Likviditet/Momentum) ---")
    shortlist_data = []
    
    # ... (LOGIKK KOMMER I NESTE STEG) ...

    shortlist_df = pd.DataFrame(shortlist_data)
    if shortlist_df.empty:
        print("Ingen aksjer kvalifisert etter Pass 1.")
        return pd.DataFrame()

    print(f"Pass 1 fullført. {len(shortlist_df)} aksjer kvalifisert for Pass 2.")

    # 3. Pass 2: Dyp Analyse (Kommer i STEG 2.3)
    print("\n--- 2. Pass 2: Dyp Analyse (V/B Simulering) ---")
    final_candidates = []
    
    # ... (LOGIKK KOMMER I NESTE STEG) ...

    final_df = pd.DataFrame(final_candidates)
    if final_df.empty: return pd.DataFrame()

    # 4. Pass 3: Normalisering og Vektet Score (Kommer i STEG 2.4)
    print("\n--- 3. Pass 3: Normalisering og Scoring ---")
    
    # ... (LOGIKK KOMMER I NESTE STEG) ...

    return final_df.head(50) # Returnerer de rangerte kandidatene

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
    # Legg til dummy-kolonner for å unngå feil ved testing, da vi ikke har fylt ut alt ennå
    if not final_result_df.empty:
        if 'asof_date' not in final_result_df.columns:
             final_result_df['asof_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    export_results(final_result_df)