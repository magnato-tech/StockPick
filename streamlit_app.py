import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- KONSTANTER ---
GITHUB_RAW_URL = "https://github.com/magnato-tech/StockPick/releases/download/latest-screener-data/top_candidates_latest.csv"

# --- KONFIGURASJON ---
st.set_page_config(page_title="StockPick | Toppkandidater", layout="wide")
st.title("🏆 StockPick: Toppkandidater for Investering")
st.markdown(
    "Kandidatene screenes ukentlig fra **S&P 1500** (S&P 500 + S&P 400 + S&P 600) "
    "og rangeres etter en vektet score av **V/B-ratio**, **Momentum** og **Volum**."
)

# --- DATAHENTING (CACHE) ---

@st.cache_data(ttl=60 * 60 * 24)
def hent_data_fra_github():
    """Henter CSV-resultatene fra GitHub Releases."""
    try:
        df = pd.read_csv(
            GITHUB_RAW_URL,
            na_values=["", " ", "NA", "N/A", "null", "None"]
        )

        if df.empty:
            return df

        numeric_cols = [
            "total_score",
            "vb_percentile", "mom_percentile", "vol_percentile",
            "sma150_bonus", "golden_cross_bonus", "vol_confirm_bonus",
            "optimal_sl_train", "cagr_test_percent", "max_drawdown_test",
            "atr_14", "dynamic_sl_pct",
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = (
                    df[c].astype(str)
                    .str.replace(",", ".", regex=False)
                    .str.replace("%", "", regex=False)
                )
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "total_score" not in df.columns or df["total_score"].dropna().empty:
            return pd.DataFrame()

        df = df.sort_values("total_score", ascending=False).reset_index(drop=True)

        # Visningskolonner
        df["V/B Score"]      = df["vb_percentile"].round(1)
        df["Mom Score"]      = df["mom_percentile"].round(1)
        df["Vol Score"]      = df["vol_percentile"].round(1)
        df["VB-SL (%)"]      = df["optimal_sl_train"].round(1)
        df["ATR-SL (%)"]     = df["dynamic_sl_pct"].round(1)
        df["CAGR Test (%)"]  = df["cagr_test_percent"].round(1)
        df["Max DD Test (%)"] = df["max_drawdown_test"].round(1)
        df["ATR(14)"]        = df["atr_14"].round(2)

        # Boolean-kolonner → lesbart symbol
        for bool_col, label in [("golden_cross", "GoldenX"), ("vol_confirmed_cross50", "VolConf")]:
            if bool_col in df.columns:
                df[label] = df[bool_col].map(
                    lambda v: "✓" if str(v).lower() in ("true", "1", "1.0") else "–"
                )

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
    st.info(
        f"Sist oppdatert: **{latest_date}** "
        f"(Run ID: {df_results['run_id'].iloc[0]}) "
        f"— Univers: **S&P 1500**"
    )

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Filtrer Resultater")

        sektor_valg      = ["Alle Sektorer"] + sorted(df_results["sector"].dropna().unique().tolist())
        selected_sector  = st.selectbox("Velg Sektor:", sektor_valg)

        num_to_show = st.slider("Vis antall kandidater:", 5, len(df_results), 20)

        st.markdown("---")
        st.subheader("Signal-filter")
        only_golden = st.checkbox("Bare Golden Cross (MA50 > MA200)", value=False)
        only_vol    = st.checkbox("Bare volum-bekreftet crossover", value=False)

        filtered_df = df_results.copy()
        if selected_sector != "Alle Sektorer":
            filtered_df = filtered_df[filtered_df["sector"] == selected_sector]
        if only_golden and "golden_cross" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["golden_cross"].astype(str).str.lower().isin(("true", "1", "1.0"))
            ]
        if only_vol and "vol_confirmed_cross50" in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df["vol_confirmed_cross50"].astype(str).str.lower().isin(("true", "1", "1.0"))
            ]

        filtered_df = filtered_df.head(num_to_show)

    st.subheader(f"Topp {len(filtered_df)} av {len(df_results)} Kandidater")

    # --- TABELL ---
    display_cols = [
        "ticker", "name", "sector",
        "total_score",
        "V/B Score", "Mom Score", "Vol Score",
        "GoldenX", "VolConf",
        "VB-SL (%)", "ATR-SL (%)", "ATR(14)",
        "CAGR Test (%)", "Max DD Test (%)",
        "why_selected",
    ]
    # Behold bare kolonner som faktisk finnes
    display_cols = [c for c in display_cols if c in filtered_df.columns]

    display_df = filtered_df[display_cols].copy()
    display_df.index = np.arange(1, len(display_df) + 1)

    st.dataframe(
        display_df.style
            .format({"total_score": "{:.1f}"})
            .background_gradient(
                cmap="RdYlGn",
                subset=[c for c in ["V/B Score", "Mom Score", "Vol Score", "total_score"] if c in display_df.columns]
            ),
        use_container_width=True,
        column_config={
            "ticker":       "Ticker",
            "name":         "Selskapsnavn",
            "sector":       "Sektor",
            "total_score":  st.column_config.NumberColumn("Total Score", help="50% V/B + 30% Mom + 20% Vol + bonuser"),
            "GoldenX":      st.column_config.TextColumn("Golden Cross", help="MA50 > MA200 (+8 poeng)"),
            "VolConf":      st.column_config.TextColumn("Vol. Bekreftet", help="Høyt volum på crossover-dagen (+5 poeng)"),
            "VB-SL (%)":    st.column_config.NumberColumn("SL Backtestet (%)", help="Optimal stop-loss fra V/B-simulering"),
            "ATR-SL (%)":   st.column_config.NumberColumn("ATR Stop-Loss (%)", help="Dynamisk SL = 2 × ATR(14) / kurs"),
            "ATR(14)":      st.column_config.NumberColumn("ATR(14)", help="Average True Range siste 14 dager"),
            "why_selected": st.column_config.TextColumn("Detaljer"),
        },
    )

    st.markdown("---")

    # --- SCOREBONUSER (expander) ---
    with st.expander("Hva betyr bonuspoengene?"):
        col1, col2, col3 = st.columns(3)
        col1.metric("SMA150-bonus", "+10 p", "Kryss opp over SMA150 nylig + fortsatt over")
        col2.metric("Golden Cross-bonus", "+8 p", "MA50 > MA200 på screeningsdagen")
        col3.metric("Volum-bekreftet crossover", "+5 p", "Volum ≥ 1.5× snitt på crossover-dagen")

    # --- GRAF ---
    st.subheader("Total Score Fordeling")

    chart_data = filtered_df.reset_index().rename(columns={"index": "Rank"})

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Rank:O", axis=None),
            y=alt.Y("total_score:Q", title="Total Score"),
            color=alt.Color("sector:N", title="Sektor"),
            tooltip=[
                "ticker", "name", "sector",
                alt.Tooltip("total_score", format=".1f"),
                "GoldenX", "VolConf",
            ],
        )
        .properties(height=300)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Total Score = vektet sum av V/B-, Momentum- og Volum-percentiler "
        "+ bonuspoeng for SMA150-kryss, Golden Cross og volum-bekreftet crossover."
    )
