import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os

# --- KONFIGURASJON ---
# Lokal CSV-fil generert av screener_motor.py
CSV_PATH = os.path.join(os.path.dirname(__file__), "top_candidates_latest.csv")

st.set_page_config(page_title="StockPick | Toppkandidater", layout="wide")
st.title("🏆 StockPick: Toppkandidater for Investering")
st.markdown(
    "Kandidatene screenes fra **S&P 1500** (S&P 500 + S&P 400 + S&P 600) "
    "og rangeres etter en vektet score av **V/B-ratio**, **Momentum** og **Volum**."
)

# --- DATAHENTING ---

@st.cache_data(ttl=60 * 5)   # 5 minutters cache lokalt (kortere enn remote-versjon)
def hent_data_lokalt():
    """Leser CSV-filen generert av screener_motor.py fra disk."""
    if not os.path.exists(CSV_PATH):
        return pd.DataFrame(), "Filen finnes ikke ennå. Kjør screener_motor.py først."

    try:
        df = pd.read_csv(CSV_PATH, na_values=["", " ", "NA", "N/A", "null", "None"])

        if df.empty:
            return pd.DataFrame(), "CSV-filen er tom. Kjør screener_motor.py på nytt."

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
            return pd.DataFrame(), "Filen mangler total_score-kolonne. Kjør screener_motor.py på nytt."

        df = df.sort_values("total_score", ascending=False).reset_index(drop=True)

        # Visningskolonner
        df["V/B Score"]       = df["vb_percentile"].round(1)
        df["Mom Score"]       = df["mom_percentile"].round(1)
        df["Vol Score"]       = df["vol_percentile"].round(1)
        df["VB-SL (%)"]       = df["optimal_sl_train"].round(1)
        df["CAGR Test (%)"]   = df["cagr_test_percent"].round(1)
        df["Max DD Test (%)"] = df["max_drawdown_test"].round(1)
        if "dynamic_sl_pct" in df.columns:
            df["ATR-SL (%)"]  = df["dynamic_sl_pct"].round(1)
        if "atr_14" in df.columns:
            df["ATR(14)"]     = df["atr_14"].round(2)

        for bool_col, label in [("golden_cross", "GoldenX"), ("vol_confirmed_cross50", "VolConf")]:
            if bool_col in df.columns:
                df[label] = df[bool_col].map(
                    lambda v: "✓" if str(v).lower() in ("true", "1", "1.0") else "–"
                )

        return df, None

    except Exception as e:
        return pd.DataFrame(), f"Feil ved lesing av CSV: {e}"


# --- LAST DATA ---
df_results, feil_melding = hent_data_lokalt()

# Knapp for å tvinge refresh (tømmer cache)
if st.button("🔄 Oppdater data"):
    st.cache_data.clear()
    st.rerun()

if df_results.empty:
    st.warning(f"Ingen data å vise. {feil_melding or ''}")
    st.info(
        "**Slik genererer du data:**\n\n"
        "Kjør følgende kommando i terminalen:\n"
        "```\n"
        f"python \"{os.path.abspath(os.path.join(os.path.dirname(__file__), 'screener_motor.py'))}\"\n"
        "```\n"
        "Screener-en tar 20–40 minutter. Trykk 'Oppdater data' når den er ferdig."
    )
else:
    latest_date = df_results["asof_date"].iloc[0] if "asof_date" in df_results.columns else "ukjent"
    mod_time    = pd.Timestamp(os.path.getmtime(CSV_PATH), unit="s").strftime("%Y-%m-%d %H:%M")
    st.success(
        f"Data fra: **{latest_date}** — Fil sist oppdatert: **{mod_time}** "
        f"— **{len(df_results)} kandidater** funnet"
    )

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Filtrer Resultater")

        sektor_valg     = ["Alle Sektorer"] + sorted(df_results["sector"].dropna().unique().tolist())
        selected_sector = st.selectbox("Velg Sektor:", sektor_valg)

        num_to_show = st.slider("Vis antall kandidater:", 5, min(50, len(df_results)), min(20, len(df_results)))

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

    with st.expander("Hva betyr bonuspoengene?"):
        col1, col2, col3 = st.columns(3)
        col1.metric("SMA150-bonus", "+10 p", "Kryss opp over SMA150 nylig + fortsatt over")
        col2.metric("Golden Cross-bonus", "+8 p", "MA50 > MA200 på screeningsdagen")
        col3.metric("Volum-bekreftet crossover", "+5 p", "Volum ≥ 1.5× snitt på crossover-dagen")

    # --- GRAF ---
    st.subheader("Total Score Fordeling")

    chart_data = filtered_df.reset_index().rename(columns={"index": "Rank"})

    chart_tooltip = ["ticker", "name", "sector", alt.Tooltip("total_score", format=".1f")]
    for col in ["GoldenX", "VolConf"]:
        if col in chart_data.columns:
            chart_tooltip.append(col)

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("Rank:O", axis=None),
            y=alt.Y("total_score:Q", title="Total Score"),
            color=alt.Color("sector:N", title="Sektor"),
            tooltip=chart_tooltip,
        )
        .properties(height=300)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Total Score = vektet sum av V/B-, Momentum- og Volum-percentiler "
        "+ bonuspoeng for SMA150-kryss, Golden Cross og volum-bekreftet crossover."
    )
