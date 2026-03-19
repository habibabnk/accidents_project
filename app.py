import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_accident_data, debug_data_directory
from modeling import train_risk_model, predict_accident_risk, get_risk_model

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RoadSight · France",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background: #070b14;
    color: #e2e8f0;
}
.stApp { background: #070b14; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-size: 0.78rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 8px !important;
    color: #a5b4fc !important;
    font-size: 0.8rem !important;
    width: 100% !important;
}

/* Page title */
.page-title {
    font-size: 2rem; font-weight: 800; line-height: 1.15;
    color: #f1f5f9; margin-bottom: 0.25rem;
}
.page-title-accent {
    background: linear-gradient(135deg, #6366f1, #a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.page-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem; color: #374151;
    text-transform: uppercase; letter-spacing: 0.14em;
    margin-bottom: 2rem;
}

/* Section label */
.section-lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; text-transform: uppercase;
    letter-spacing: 0.16em; color: #6366f1;
    margin: 2rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.15);
}

/* Warning */
.warn {
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px; padding: 0.75rem 1rem;
    font-size: 0.82rem; color: #fbbf24;
    margin-bottom: 1.5rem;
}
.err {
    background: rgba(239,68,68,0.07);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 10px; padding: 0.75rem 1rem;
    font-size: 0.82rem; color: #f87171;
}
.ok {
    background: rgba(16,185,129,0.07);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 10px; padding: 0.6rem 1rem;
    font-size: 0.8rem; color: #34d399;
    font-family: 'JetBrains Mono', monospace;
}

/* Primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important; border-radius: 10px !important;
    color: white !important; font-weight: 600 !important;
    padding: 0.55rem 1.8rem !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.28) !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 8px !important;
}
.stSelectbox label { color: #6b7280 !important; font-size: 0.78rem !important; }

/* Slider */
.stSlider label { color: #6b7280 !important; font-size: 0.78rem !important; }

/* Logo */
.logo-wrap {
    padding: 0.25rem 0 1.75rem 0;
}
.logo-name {
    font-size: 1.05rem; font-weight: 800;
    background: linear-gradient(90deg, #a5b4fc, #f0abfc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.logo-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem; color: #374151;
    text-transform: uppercase; letter-spacing: 0.12em;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Chart base ────────────────────────────────────────────────────────────────
COLORS = ["#6366f1", "#a855f7", "#ec4899", "#10b981", "#f59e0b", "#3b82f6", "#14b8a6"]

BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#64748b", size=11),
    margin=dict(l=8, r=8, t=32, b=8),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.05)",
               tickfont=dict(size=10, color="#4b5563")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", linecolor="rgba(255,255,255,0.05)",
               tickfont=dict(size=10, color="#4b5563")),
    hoverlabel=dict(bgcolor="#111827", bordercolor="#1f2937",
                    font=dict(family="JetBrains Mono", size=11, color="#e2e8f0")),
    colorway=COLORS,
)

def apply(fig, h=340, title=""):
    fig.update_layout(**BASE, height=h,
                      title=dict(text=title, font=dict(size=11, color="#4b5563"),
                                 x=0, xanchor="left", pad=dict(b=8)) if title else {})
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────
def section(text):
    st.markdown(f'<div class="section-lbl">{text}</div>', unsafe_allow_html=True)

def header(plain, accent, sub):
    st.markdown(
        f'<div class="page-title">{plain} '
        f'<span class="page-title-accent">{accent}</span></div>'
        f'<div class="page-sub">{sub}</div>',
        unsafe_allow_html=True
    )


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_overview(df):
    header("Tableau de", "Bord", "France · 2015 – 2024 · Vue d'ensemble")

    if df is None or len(df) == 0:
        st.markdown('<div class="err">Aucune donnée chargée.</div>', unsafe_allow_html=True)
        return

    # KPI row — using st.columns (no unsafe HTML)
    n_acc   = len(df)
    n_fatal = int(df["fatalities"].fillna(0).sum()) if "fatalities" in df.columns else 0
    n_ser   = int(df["serious_injuries"].fillna(0).sum()) if "serious_injuries" in df.columns else 0
    rate    = f"{df['is_serious'].mean()*100:.1f}%" if "is_serious" in df.columns else "—"
    yr_cov  = f"{int(df['year'].min())} – {int(df['year'].max())}" if "year" in df.columns else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val, color in [
        (c1, "Accidents recensés", f"{n_acc:,}",   "#a5b4fc"),
        (c2, "Décès",              f"{n_fatal:,}",  "#fca5a5"),
        (c3, "Blessés graves",     f"{n_ser:,}",    "#fcd34d"),
        (c4, "Taux de gravité",    rate,            "#6ee7b7"),
        (c5, "Couverture",         yr_cov,          "#94a3b8"),
    ]:
        with col:
            st.markdown(
                f"""<div style="background:rgba(255,255,255,0.02);border:1px solid
                rgba(255,255,255,0.07);border-radius:14px;padding:1.1rem 1.2rem;
                border-top:2px solid {color}30">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                text-transform:uppercase;letter-spacing:0.12em;color:#4b5563;
                margin-bottom:0.4rem">{label}</div>
                <div style="font-size:1.6rem;font-weight:800;color:{color};
                line-height:1">{val}</div></div>""",
                unsafe_allow_html=True
            )

    # Trend
    if "year" in df.columns:
        section("Evolution annuelle")
        yt = df.groupby("year").size().reset_index(name="n")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yt["year"], y=yt["n"], mode="none",
            fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=yt["year"], y=yt["n"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=2.5, shape="spline"),
            marker=dict(size=7, color="#6366f1", line=dict(width=2, color="#070b14")),
            name="Accidents",
            hovertemplate="<b>%{x}</b> — %{y:,} accidents<extra></extra>",
        ))
        if 2020 in yt["year"].values:
            fig.add_vline(x=2020, line_dash="dot", line_color="rgba(245,158,11,0.4)",
                          annotation_text="COVID-19",
                          annotation_font=dict(color="#f59e0b", size=10),
                          annotation_position="top right")
        apply(fig, 260)
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    if "hour" in df.columns and "day_of_week" in df.columns:
        section("Densite temporelle · Heure x Jour")
        days = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
        heat  = df.groupby(["day_of_week", "hour"]).size().reset_index(name="n")
        pivot = heat.pivot(index="day_of_week", columns="hour", values="n").fillna(0)
        pivot.index = [days[i] for i in pivot.index if i < 7]
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}h" for h in pivot.columns],
            y=list(pivot.index),
            colorscale=[
                [0.0, "rgba(7,11,20,1)"],
                [0.4, "rgba(99,102,241,0.4)"],
                [0.75, "rgba(168,85,247,0.75)"],
                [1.0, "rgba(236,72,153,1)"],
            ],
            showscale=True,
            colorbar=dict(tickfont=dict(color="#4b5563", size=9),
                          outlinewidth=0, thickness=10),
            hovertemplate="%{y} · %{x}<br><b>%{z:.0f}</b> accidents<extra></extra>",
        ))
        apply(fig, 250)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        if "department" in df.columns:
            section("Top 15 departements")
            td = df["department"].value_counts().head(15).reset_index()
            td.columns = ["dept", "n"]
            td = td.sort_values("n")
            fig = go.Figure(go.Bar(
                x=td["n"], y=td["dept"].astype(str), orientation="h",
                marker=dict(
                    color=td["n"],
                    colorscale=[[0, "rgba(99,102,241,0.15)"], [1, "#6366f1"]],
                    line=dict(width=0),
                ),
                hovertemplate="Dep. %{y}<br><b>%{x:,}</b> accidents<extra></extra>",
            ))
            apply(fig, 420)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "weather" in df.columns:
            section("Conditions meteo")
            w = df["weather"].value_counts().head(7)
            fig = go.Figure(go.Pie(
                values=w.values, labels=w.index, hole=0.6,
                marker=dict(colors=COLORS, line=dict(color="#070b14", width=3)),
                hovertemplate="%{label}<br><b>%{value:,}</b> · %{percent}<extra></extra>",
                textfont=dict(size=10, color="#94a3b8"),
                textposition="outside",
            ))
            apply(fig, 420)
            fig.update_layout(
                legend=dict(font=dict(size=9, color="#4b5563"), bgcolor="rgba(0,0,0,0)"),
                annotations=[dict(text="Meteo", x=0.5, y=0.5,
                                  font=dict(size=12, color="#64748b",
                                            family="JetBrains Mono"),
                                  showarrow=False)],
            )
            st.plotly_chart(fig, use_container_width=True)

    if "hour" in df.columns:
        section("Accidents par heure")
        h = df["hour"].value_counts().sort_index().reset_index()
        h.columns = ["hour", "n"]
        fig = go.Figure(go.Bar(
            x=h["hour"], y=h["n"],
            marker=dict(
                color=h["n"],
                colorscale=[[0, "rgba(99,102,241,0.15)"], [1, "rgba(236,72,153,0.85)"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}h</b> — %{y:,} accidents<extra></extra>",
        ))
        apply(fig, 220)
        st.plotly_chart(fig, use_container_width=True)


def page_filters(df):
    header("Exploration", "Filtree", "Filtres dynamiques · Analyse personnalisee")

    if df is None or len(df) == 0:
        st.markdown('<div class="err">Aucune donnee chargee.</div>', unsafe_allow_html=True)
        return

    fdf = df.copy()

    with st.sidebar:
        section("Filtres")
        if "year" in df.columns:
            yr = st.slider("Annees", int(df["year"].min()), int(df["year"].max()),
                           (int(df["year"].min()), int(df["year"].max())))
            fdf = fdf[(fdf["year"] >= yr[0]) & (fdf["year"] <= yr[1])]
        if "hour" in df.columns:
            hr = st.slider("Heures", 0, 23, (0, 23))
            fdf = fdf[(fdf["hour"] >= hr[0]) & (fdf["hour"] <= hr[1])]
        if "department" in df.columns:
            depts = sorted(df["department"].dropna().unique())
            sel_d = st.multiselect("Departements", depts, default=depts[:15])
            if sel_d:
                fdf = fdf[fdf["department"].isin(sel_d)]
        if "weather" in df.columns:
            weathers = sorted(df["weather"].dropna().unique())
            sel_w = st.multiselect("Meteo", weathers, default=weathers)
            if sel_w:
                fdf = fdf[fdf["weather"].isin(sel_w)]

    pct  = len(fdf) / len(df) * 100 if len(df) else 0
    rate = f"{fdf['is_serious'].mean()*100:.1f}%" if "is_serious" in fdf.columns else "—"
    fatal = int(fdf["fatalities"].fillna(0).sum()) if "fatalities" in fdf.columns else 0

    c1, c2, c3 = st.columns(3)
    for col, label, val, color in [
        (c1, "Accidents filtres", f"{len(fdf):,}",  "#a5b4fc"),
        (c2, "Taux de gravite",   rate,              "#6ee7b7"),
        (c3, "Deces",             f"{fatal:,}",      "#fca5a5"),
    ]:
        with col:
            st.markdown(
                f"""<div style="background:rgba(255,255,255,0.02);border:1px solid
                rgba(255,255,255,0.07);border-radius:14px;padding:1.1rem 1.2rem;
                border-top:2px solid {color}30">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                text-transform:uppercase;letter-spacing:0.12em;color:#4b5563;
                margin-bottom:0.4rem">{label}</div>
                <div style="font-size:1.6rem;font-weight:800;color:{color};
                line-height:1">{val}</div>
                <div style="font-size:0.7rem;color:#374151;font-family:'JetBrains Mono',
                monospace;margin-top:0.25rem">{pct:.1f}% du total</div></div>""",
                unsafe_allow_html=True
            )

    col1, col2 = st.columns(2)
    with col1:
        if "department" in fdf.columns:
            section("Departements")
            d = fdf["department"].value_counts().head(12).reset_index()
            d.columns = ["dept", "n"]
            d = d.sort_values("n")
            fig = go.Figure(go.Bar(
                x=d["n"], y=d["dept"].astype(str), orientation="h",
                marker=dict(color="#6366f1", line=dict(width=0), opacity=0.8),
                hovertemplate="Dep. %{y}<br><b>%{x:,}</b><extra></extra>",
            ))
            apply(fig, 360)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "hour" in fdf.columns:
            section("Distribution horaire")
            h = fdf["hour"].value_counts().sort_index().reset_index()
            h.columns = ["hour", "n"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=h["hour"], y=h["n"], mode="lines+markers",
                line=dict(color="#ec4899", width=2, shape="spline"),
                fill="tozeroy", fillcolor="rgba(236,72,153,0.07)",
                marker=dict(size=5, color="#ec4899"),
                hovertemplate="<b>%{x}h</b> — %{y:,}<extra></extra>",
            ))
            apply(fig, 360)
            st.plotly_chart(fig, use_container_width=True)

    if "year" in fdf.columns:
        section("Tendance annuelle")
        yt = fdf.groupby("year").size().reset_index(name="n")
        fig = go.Figure(go.Bar(
            x=yt["year"], y=yt["n"],
            marker=dict(
                color=yt["n"],
                colorscale=[[0, "rgba(99,102,241,0.15)"], [1, "#6366f1"]],
                line=dict(width=0),
            ),
            hovertemplate="<b>%{x}</b> — %{y:,}<extra></extra>",
        ))
        apply(fig, 220)
        st.plotly_chart(fig, use_container_width=True)

    section("Apercu des donnees")
    st.dataframe(fdf.head(500), use_container_width=True, height=260)


def page_prediction(df):
    header("Prediction de", "Risque", "Modele Random Forest · Estimation statistique")

    st.markdown(
        '<div class="warn">Outil statistique base sur des donnees historiques. '
        'Ne pas utiliser pour des decisions de securite en temps reel.</div>',
        unsafe_allow_html=True
    )

    if df is None or len(df) == 0:
        st.markdown('<div class="err">Aucune donnee disponible.</div>', unsafe_allow_html=True)
        return

    # ── Debug target distribution ─────────────────────────────────────────────
    if "is_serious" in df.columns:
        dist = df["is_serious"].value_counts()
        if df["is_serious"].nunique() < 2:
            st.markdown(
                f'<div class="err">La colonne cible "is_serious" ne contient qu\'une seule valeur '
                f'({dist.index[0]}). Verifiez que vos CSV contiennent bien la colonne "grav" '
                f'avec des valeurs >= 3 (blesses graves / deces).</div>',
                unsafe_allow_html=True
            )
            with st.expander("Debug — colonnes disponibles"):
                st.write(list(df.columns))
                st.write(df.head(3))
            return

    model = get_risk_model()

    if not model.is_trained:
        section("Entrainement du modele")
        if st.button("Entrainer le modele Random Forest", type="primary"):
            with st.spinner("Entrainement en cours..."):
                success, result, trained_model = train_risk_model(df)
            if success:
                m = result
                st.markdown('<div class="ok">Modele entraine avec succes</div>',
                            unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                for col, label, val in [
                    (c1, "Accuracy", f"{m['accuracy']:.3f}"),
                    (c2, "AUC Score", f"{m['auc']:.3f}"),
                ]:
                    with col:
                        st.markdown(
                            f"""<div style="background:rgba(99,102,241,0.06);border:1px solid
                            rgba(99,102,241,0.2);border-radius:12px;padding:1rem 1.2rem">
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                            text-transform:uppercase;letter-spacing:0.12em;color:#4b5563;
                            margin-bottom:0.3rem">{label}</div>
                            <div style="font-size:1.8rem;font-weight:800;color:#a5b4fc">
                            {val}</div></div>""",
                            unsafe_allow_html=True
                        )

                if trained_model and trained_model.get_feature_importance() is not None:
                    section("Importance des variables")
                    imp = trained_model.get_feature_importance().head(12).sort_values("importance")
                    fig = go.Figure(go.Bar(
                        x=imp["importance"], y=imp["feature"], orientation="h",
                        marker=dict(
                            color=imp["importance"],
                            colorscale=[[0, "rgba(99,102,241,0.15)"], [1, "#a855f7"]],
                            line=dict(width=0),
                        ),
                        hovertemplate="%{y}<br><b>%{x:.4f}</b><extra></extra>",
                    ))
                    apply(fig, 380)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(f'<div class="err">Echec : {result}</div>',
                            unsafe_allow_html=True)
        return

    # ── Form ──────────────────────────────────────────────────────────────────
    section("Parametres de prediction")

    c1, c2, c3 = st.columns(3)
    with c1:
        dept_opts = sorted(df["department"].dropna().unique()) if "department" in df.columns else ["75"]
        sel_dept  = st.selectbox("Departement", dept_opts)
        sel_month = st.selectbox("Mois", range(1, 13),
                                  format_func=lambda x: ["Jan","Fev","Mar","Avr","Mai","Jun",
                                                          "Jul","Aou","Sep","Oct","Nov","Dec"][x-1])
    with c2:
        sel_dow  = st.selectbox("Jour", range(7),
                                 format_func=lambda x: ["Lun","Mar","Mer","Jeu","Ven","Sam","Dim"][x])
        sel_hour = st.selectbox("Heure", range(24), format_func=lambda x: f"{x:02d}:00")
    with c3:
        weather_opts  = sorted(df["weather"].dropna().unique()) if "weather" in df.columns else ["Normal"]
        lighting_opts = sorted(df["lighting"].dropna().unique()) if "lighting" in df.columns else ["Jour"]
        sel_weather  = st.selectbox("Meteo", weather_opts)
        sel_lighting = st.selectbox("Eclairage", lighting_opts)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Calculer le risque", type="primary"):
        try:
            pred = predict_accident_risk({
                "year": 2024, "month": sel_month, "day_of_week": sel_dow,
                "hour": sel_hour, "department": sel_dept,
                "weather": sel_weather, "lighting": sel_lighting,
            })
            rp = pred["risk_percentage"]

            risk_color = "#10b981" if rp < 20 else "#f59e0b" if rp < 40 else "#ef4444"
            risk_label = "Faible" if rp < 20 else "Modere" if rp < 40 else "Eleve"

            section("Resultat")

            cg, cm = st.columns([1, 1])
            with cg:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rp,
                    number=dict(suffix="%",
                                font=dict(size=44, color=risk_color,
                                          family="Plus Jakarta Sans")),
                    gauge=dict(
                        axis=dict(range=[0, 100],
                                  tickcolor="rgba(255,255,255,0.08)",
                                  tickfont=dict(color="#374151", size=9)),
                        bar=dict(color=risk_color, thickness=0.22),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 20],   color="rgba(16,185,129,0.07)"),
                            dict(range=[20, 40],  color="rgba(245,158,11,0.07)"),
                            dict(range=[40, 100], color="rgba(239,68,68,0.07)"),
                        ],
                        threshold=dict(
                            line=dict(color=risk_color, width=2),
                            thickness=0.75, value=rp,
                        ),
                    ),
                    domain=dict(x=[0, 1], y=[0, 1]),
                ))
                fig.update_layout(**BASE, height=280, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with cm:
                st.markdown("<br>", unsafe_allow_html=True)
                for label, val, color in [
                    ("Probabilite d'accident grave", f"{rp:.1f}%", "#a5b4fc"),
                    ("Niveau de risque", risk_label, risk_color),
                ]:
                    st.markdown(
                        f"""<div style="background:rgba(255,255,255,0.02);border:1px solid
                        rgba(255,255,255,0.06);border-radius:12px;padding:1rem 1.2rem;
                        margin-bottom:0.75rem;border-left:3px solid {color}">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                        text-transform:uppercase;letter-spacing:0.12em;color:#4b5563;
                        margin-bottom:0.3rem">{label}</div>
                        <div style="font-size:1.6rem;font-weight:800;color:{color}">{val}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.markdown(f'<div class="err">Erreur : {e}</div>', unsafe_allow_html=True)


def page_about():
    header("A", "Propos", "Methodologie · Sources · Limites")
    st.markdown("""
<div style="max-width:680px;line-height:1.9;color:#64748b;font-size:0.88rem">
<p><strong style="color:#94a3b8">Source.</strong>
Base nationale d'accidentologie routiere francaise 2015-2024, Ministere de l'Interieur.
Les tables caracteristiques, usagers, lieux et vehicules sont fusionnees pour reconstituer
le profil complet de chaque accident.</p>

<p style="margin-top:1rem"><strong style="color:#94a3b8">Pipeline.</strong>
Chargement multi-format (separateurs ; / , , encodages UTF-8/Latin-1), standardisation
des colonnes, extraction temporelle, fusion des niveaux de gravite (grav >= 3 = accident grave).
Split temporel strict : entrainement 2019-2023, evaluation 2024.</p>

<p style="margin-top:1rem"><strong style="color:#94a3b8">Modele.</strong>
Random Forest (100 arbres, profondeur max 10, poids equilibres entre classes).
Variables : mois, heure, jour de la semaine, departement, meteo, eclairage.</p>

<p style="margin-top:1rem"><strong style="color:#94a3b8">Limites.</strong>
Estimations statistiques sur donnees historiques uniquement. Usage analytique et
pedagogique — ne pas utiliser pour des decisions de securite en temps reel.</p>

<p style="margin-top:1rem"><strong style="color:#94a3b8">Stack.</strong>
Python · Streamlit · Pandas · Scikit-learn · Plotly</p>
</div>
""", unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with st.sidebar:
        st.markdown("""
        <div class="logo-wrap">
            <div class="logo-name">RoadSight</div>
            <div class="logo-sub">France · Analytics</div>
        </div>
        """, unsafe_allow_html=True)

        default_dir = str(Path(__file__).parent)
        data_dir = st.text_input(
            "data_dir", value=default_dir, label_visibility="collapsed",
            placeholder="Chemin vers le dossier donnees"
        )

        if st.button("Scanner le dossier"):
            n, preview = debug_data_directory(data_dir)
            if n:
                st.markdown(f'<div class="ok">{n} fichiers CSV trouves</div>',
                            unsafe_allow_html=True)
                for p in preview[:6]:
                    st.caption(p)
            else:
                st.markdown('<div class="err">Aucun CSV trouve</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Navigation")

        page = st.radio("nav", [
            "Vue d'ensemble",
            "Exploration filtree",
            "Prediction de risque",
            "A propos",
        ], label_visibility="collapsed")

    # Load
    df, errors = None, []
    with st.spinner("Chargement..."):
        try:
            result = load_accident_data(data_dir)
            df, errors = result if isinstance(result, tuple) else (result, [])
        except Exception as e:
            st.error(f"Erreur critique : {e}")

    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        if df is not None and len(df) > 0:
            st.markdown(
                f'<div class="ok">{len(df):,} enregistrements charges</div>',
                unsafe_allow_html=True
            )
        elif errors:
            st.markdown(
                f'<div class="err">{len(errors)} erreur(s) de chargement</div>',
                unsafe_allow_html=True
            )
            with st.expander("Details"):
                for e in errors[:3]:
                    st.caption(f"{Path(e['file']).name}: {e['error'][:55]}")

    if page == "Vue d'ensemble":
        page_overview(df)
    elif page == "Exploration filtree":
        page_filters(df)
    elif page == "Prediction de risque":
        page_prediction(df)
    else:
        page_about()


if __name__ == "__main__":
    main()