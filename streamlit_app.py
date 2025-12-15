import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- KONSTANTER ---
# Denne URLen peker direkte på 'latest-screener-data' Release Asset i ditt GitHub-repo.
# Filen oppdateres automatisk hver uke av GitHub Actions.
GITHUB_RAW_URL = "https://github.com/magnato-tech/StockPick/releases/download/latest-screener-data/top_candidates_latest.csv"

# --- KONFIGURASJON ---
st.set_page_config(page_title="StockPick | Toppkandidater", layout="wide")
st.title("🏆 StockPick: Toppkandidater for Investering")
st.markdown("Denne listen er generert ukentlig basert på vår **robuste kvantitative screening-motor** (Pass 0-3).")
st.markdown("Kandidatene er rangert etter en vektet score av **V/B-ratio**, **Momentum** og **Volum.**")

# --- DATAHENTING (CACHE) ---

@st.cache_data(ttl=60 * 60 * 24)
def hent_data_fra_github():
    """Henter de ferdige CSV-resultatene fra GitHub Releases (robust mot tom/objekt-data)."""
    try:
        df = pd.read_csv(
            GITHUB_RAW_URL,
            na_values=["", " ", "NA", "N/A", "null", "None"]
        )

        # Hvis fila er tom (headers only / 0 rader), ikke prøv sort/round
        if df.empty:
            return df

        # Tving numeriske kolonner til numeric (håndterer også komma/prosent om det skulle forekomme)
        numeric_cols = [
            "total_score",
            "vb_percentile",
            "mom_percentile",
            "vol_percentile",
            "optimal_sl_train",
            "cagr_test_percent",
            "max_drawdown_test",
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = (
                    df[c]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .str.replace("%", "", regex=False)
                )
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Hvis total_score mangler eller ikke ble numerisk -> ingen gyldige data å vise
        if "total_score" not in df.columns or df["total_score"].dropna().empty:
            return pd.DataFrame()

        # Sorterer dataen på nytt, bare for å være sikker
        df = df.sort_values(by="total_score", ascending=False).reset_index(drop=True)

        # Formaterer visningskolonner (nå er de garantert numeriske eller NaN)
        df["V/B Score"] = df["vb_percentile"].round(1)
        df["Momentum Score"] = df["mom_percentile"].round(1)
        df["Volum Score"] = df["vol_percentile"].round(1)
        df["Optimal SL (%)"] = df["optimal_sl_train"].round(1)
        df["CAGR Test (%)"] = df["cagr_test_percent"].round(1)
        df["Max DD Test (%)"] = df["max_drawdown_test"].round(1)

        return df

    except Exception as e:
        st.error(f"Klarte ikke å laste data fra GitHub Releases: {e}")
        return pd.DataFrame()

# --- HOVEDAPPLOGIKK ---

df_results = hent_data_fra_github()

if df_results.empty:
    st.warning("Kan ikke vise resultater akkurat nå. Sjekk at backend-pipelinen har kjørt.")
else:
    latest_date = df_results["asof_date"].iloc[0]
    st.info(f"Sist oppdatert: **{latest_date}** (Kjørt fra Run ID: {df_results['run_id'].iloc[0]})")

    # Filtreringspanel (Sidebar)
    with st.sidebar:
        st.header("Filtrer Resultater")
        sektor_valg = ["Alle Sektorer"] + sorted(df_results["sector"].dropna().unique().tolist())
        selected_sector = st.selectbox("Velg Sektor:", sektor_valg)

        num_to_show = st.slider("Vis antall kandidater:", 5, len(df_results), 20)

        filtered_df = df_results.copy()
        if selected_sector != "Alle Sektorer":
            filtered_df = filtered_df[filtered_df["sector"] == selected_sector]

        filtered_df = filtered_df.head(num_to_show)

    st.subheader(f"Topp {len(filtered_df)} av {len(df_results)} Kandidater")

    # --- VISNING AV RESULTAT (Tabell) ---

    display_cols = [
        "ticker", "name", "sector",
        "V/B Score", "Momentum Score", "Volum Score", "total_score",
        "Optimal SL (%)", "CAGR Test (%)", "Max DD Test (%)", "why_selected"
    ]

    # Gi rangeringsnummer (kun for visning)
    display_df = filtered_df[display_cols].copy()
    display_df.index = np.arange(1, len(display_df) + 1)

    st.dataframe(
        display_df.style
            .format({"total_score": "{:.1f}"})
            .background_gradient(cmap="RdYlGn", subset=["V/B Score", "Momentum Score", "Volum Score", "total_score"]),
        use_container_width=True,
        column_order=[
            "ticker", "name", "sector", "total_score",
            "V/B Score", "Momentum Score", "Volum Score",
            "Optimal SL (%)", "CAGR Test (%)", "Max DD Test (%)", "why_selected"
        ],
        column_config={
            "ticker": "Ticker",
            "name": "Selskapsnavn",
            "sector": "Sektor",
            "total_score": st.column_config.NumberColumn(
                "Total Score",
                help="Vektet Score (50% V/B + 30% Mom + 20% Vol)"
            ),
            "why_selected": st.column_config.TextColumn("Detaljer"),
            "Optimal SL (%)": st.column_config.NumberColumn(
                "Optimal SL (Train)",
                help="SL-prosent optimalisert på Train-settet"
            ),
        }
    )

    st.markdown("---")

    # --- GRAF OVER TOTAL SCORE ---
    st.subheader("Total Score Fordeling")

    chart_data = filtered_df.reset_index().rename(columns={"index": "Rank"})

    chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("Rank:O", axis=None),
        y=alt.Y("total_score:Q", title="Total Score"),
        color=alt.Color("sector:N", title="Sektor"),
        tooltip=["ticker", "name", "sector", alt.Tooltip("total_score", format=".1f")]
    ).properties(
        height=300
    ).interactive()

    st.altair_chart(chart, use_container_width=True)
    st.caption("Total Score er vektet sum av V/B, Momentum og Volum percentiler.")
