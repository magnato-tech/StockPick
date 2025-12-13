import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import requests
import io
import time

# --- KONFIGURASJON ---
st.set_page_config(page_title="Stop Loss Optimalisering", layout="wide")
st.title("📈 Aksjeanalyse: Stop Loss Optimalisering")
st.markdown("Her analyseres daglige bevegelser (OHLC) for å optimalisere Stop Loss (SL) nivået.")

# --- DATACONFIG OG FUNKSJONER ---

@st.cache_data(ttl=60*60*24)
def hent_data(ticker, start, end):
    """Henter historiske aksjedata med 24-timers cache."""
    try:
        # Henter daglige data
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

# Funksjonen for Ticker-søk, nå med 24-timers cache
@st.cache_data(ttl=60*60*24)
def finn_ticker_fra_navn(navn_eller_ticker):
    """Finner en Ticker basert på navn (eller returnerer input hvis det er en ticker)."""
    if navn_eller_ticker.strip().isupper() and ('.' in navn_eller_ticker or len(navn_eller_ticker) <= 5):
        return navn_eller_ticker.upper()
    
    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = "Mozilla/5.0"
        params = {"q": navn_eller_ticker, "quotes_count": 1}
        res = requests.get(url=url, params=params, headers={'User-Agent': user_agent}, timeout=5)
        res.raise_for_status()
        data = res.json()

        if data and 'quotes' in data and len(data['quotes']) > 0:
            beste_match = data['quotes'][0]
            ticker = beste_match.get('symbol')
            exchange = beste_match.get('exchange')
            
            # Spesifikt for Oslo Børs
            if exchange == 'OSL':
                return f"{ticker}.OL"
            
            return ticker
        
    except requests.exceptions.RequestException:
        return None
    return None

def simuler_handel(df, kjops_dato, kjops_pris, stop_loss_pct):
    """Simulerer en Trailing Stop Loss handel og returnerer gevinst."""
    periode_data = df[df.index > kjops_dato].copy()
    
    if periode_data.empty: return 0.0, kjops_dato, kjops_pris

    hoyeste_pris = kjops_pris
    
    for dato, row in periode_data.iterrows():
        if row['High'] > hoyeste_pris:
            hoyeste_pris = row['High']
            
        stop_niva = hoyeste_pris * (1 - stop_loss_pct)
        
        # Sjekk om Low treffer stop loss (Stop Loss trigges)
        if row['Low'] <= stop_niva:
            salgspris = stop_niva
            gevinst = salgspris - kjops_pris
            gevinst_pct = gevinst / kjops_pris
            return gevinst_pct, dato, salgspris

    # Hvis ikke stoppet ut
    siste_pris = periode_data.iloc[-1]['Close']
    gevinst = siste_pris - kjops_pris
    gevinst_pct = gevinst / kjops_pris
    return gevinst_pct, periode_data.index[-1], siste_pris


# --- SIDEBAR (INPUT) ---
with st.sidebar:
    st.header("Innstillinger")
    input_aksje = st.text_input("Selskapsnavn eller Ticker", value="EQNR.OL")
    
    default_start = pd.to_datetime("today") - pd.DateOffset(years=2)
    start_date = st.date_input("Startdato", value=default_start)
    end_date = st.date_input("Sluttdato", value=pd.to_datetime("today"))
    
    st.markdown("---")
    stop_loss_range = st.slider("Test Stop Loss fra/til %", 1, 90, (5, 30))
    
    kjør_knapp = st.button("Kjør Analyse")


# --- HOVEDLOGIKK ---

if kjør_knapp:
    # 1. Finn Ticker
    with st.spinner('Søker etter ticker...'):
        funnet_ticker = finn_ticker_fra_navn(input_aksje)

    if not funnet_ticker:
        st.error(f"Fant ingen gyldig ticker for '{input_aksje}'.")
    else:
        st.info(f"Fant Ticker: **{funnet_ticker}**")

        # 2. Hent Data
        with st.spinner(f'Henter data for {funnet_ticker}...'):
            df = hent_data(funnet_ticker, start_date, end_date)
            
        if df is None or df.empty:
            st.error(f"Fant ingen historisk data for ticker **{funnet_ticker}** i perioden.")
        else:
            
            # Identifiser Kjøpspunkt (Laveste kurs i perioden)
            min_row = df.loc[df['Low'].idxmin()]
            optimal_dato = min_row.name
            optimal_pris = min_row['Low']
            
            # --- SIMULERINGSLØKKE ---
            best_gevinst = -100.0
            best_sl = 0
            best_salgsdato = None
            best_salgspris = 0
            results = []
            
            r_start, r_end = stop_loss_range
            range_sl = range(r_start, r_end + 1)
            
            my_bar = st.progress(0, text="Kjører Stop Loss Grid Search...")
            
            for i, sl in enumerate(range_sl):
                sl_desimal = sl / 100.0
                g, s_dato, s_pris = simuler_handel(df, optimal_dato, optimal_pris, sl_desimal)
                results.append({'Stop Loss %': sl, 'Gevinst %': g*100})
                
                if g > best_gevinst:
                    best_gevinst = g
                    best_sl = sl
                    best_salgsdato = s_dato
                    best_salgspris = s_pris
                
                my_bar.progress((i + 1) / len(range_sl))
            
            my_bar.empty()

            # --- SEKSJON 1: KPI-er ---
            st.markdown(f"### 📊 Resultater for {funnet_ticker}")
            
            dato_kjop = optimal_dato.strftime('%d.%m.%Y')
            dato_salg = f"{best_salgsdato.strftime('%d.%m.%Y')}"
            salgsbeskrivelse = "I dag" if best_salgsdato == df.index[-1] else "Stop Loss utløst"

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Kjøpspris", f"{optimal_pris:.2f} kr")
                st.caption(f"📅 {dato_kjop}")
            with col2:
                st.metric("Salgspris", f"{best_salgspris:.2f} kr")
                st.caption(f"📅 {dato_salg} ({salgsbeskrivelse})")
            with col3:
                farge = "normal" if best_gevinst > 0 else "inverse"
                st.metric("Total Gevinst", f"{best_gevinst*100:.2f} %", delta_color=farge)
            with col4:
                st.metric("Optimal Stop Loss", f"{best_sl} %")

            st.markdown("---")

            # --- SEKSJON 2: GRAFER (Viser hele perioden) ---
            plot_data = df.copy() 
            hoyeste = optimal_pris
            sl_line = []
            
            # Beregn SL-linjen for hele perioden (NaN før kjøp)
            for dato in plot_data.index:
                if dato < optimal_dato:
                    sl_line.append(np.nan)
                    continue
                
                row = plot_data.loc[dato]
                if row['High'] > hoyeste: hoyeste = row['High']
                    
                stop_loss_value = hoyeste * (1 - (best_sl/100.0))
                sl_line.append(stop_loss_value)
            
            plot_data['Stop Loss Linje'] = sl_line
            
            st.subheader("Kursutvikling vs. Optimal Stop Loss")
            
            base = alt.Chart(plot_data.reset_index()).encode(x=alt.X('Date:T', title='Dato'))
            line_close = base.mark_line(color='#1f77b4').encode(y=alt.Y('Close:Q', title='Kurs (kr)'))
            line_sl = base.mark_line(color='red', strokeDash=[5,5]).encode(y='Stop Loss Linje:Q')
            
            # Markør for kjøp
            buy_df = pd.DataFrame({'Date': [optimal_dato], 'Price': [optimal_pris]})
            buy_point = alt.Chart(buy_df).mark_point(color='green', size=150, filled=True, shape='triangle-up').encode(
                x='Date:T', y='Price:Q'
            )
            
            # Markør for salg
            exit_df = pd.DataFrame({'Date': [best_salgsdato], 'Price': [best_salgspris]})
            exit_point = alt.Chart(exit_df).mark_point(color='orange', size=150, filled=True).encode(
                x='Date:T', y='Price:Q'
            )

            st.altair_chart(line_close + line_sl + buy_point + exit_point, use_container_width=True)
            st.caption("Blå linje: Sluttkurs. Rød stiplet: Optimal SL. Grønn: Kjøp. Oransje: Salg.")
            
            # --- Resultat-tabell ---
            res_df = pd.DataFrame(results)
            with st.expander("Se Stop Loss Grid Search Resultater"):
                st.dataframe(res_df.style.highlight_max(axis=0, subset=['Gevinst %'], color='lightgreen'), use_container_width=True)


# Kjør Streamlit lokalt:
# streamlit run streamlit_app.py

# --- SLUTT PÅ KODE ---

**Spørsmål til verifisering:**

1.  Er koden lagt inn i `streamlit_app.py`?
2.  Kan dere kjøre appen lokalt (`streamlit run streamlit_app.py`) og bekrefte at dere kan søke med både ticker (`EQNR.OL`) og selskapsnavn (`Equinor`)?
3.  Bekreft at Stop Loss-linjen vises for **hele perioden** (med oransje prikk der SL trigget), og ikke stopper når SL krysses.