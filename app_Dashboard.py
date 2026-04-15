import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="PEREN AI – Digital Twin")

# ====================== STYLE CSS GLOBAL ======================
st.markdown("""
<style>
    .stApp {
        background-color: #0a0a0a !important;
        color: #e0f2e9;
    }
    .user-card {
        background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
        padding: 25px;
        border-radius: 16px;
        border: 1px solid #10b981;
        margin-bottom: 25px;
    }
    .initials {
        background: #10b981;
        color: #0a0a0a;
        font-weight: bold;
        font-size: 32px;
        width: 70px;
        height: 70px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        float: left;
        margin-right: 20px;
    }
    /* Cartes Health Indicators - même taille */
    .health-card {
        background: #1a1a1a;
        padding: 30px;
        border-radius: 16px;
        border: 1px solid;
        height: 320px !important;
        display: flex;
        flex-direction: column;
    }
    /* Tous les boutons en noir */
    .stButton>button {
        background-color: #1a1a1a !important;
        color: #e0f2e9 !important;
        border: 1px solid #333333 !important;
        border-radius: 8px;
        font-weight: 500;
        height: 45px;
    }
    .stButton>button:hover {
        background-color: #2a2a2a !important;
        border-color: #10b981 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("PEREN AI – Digital Twin Dashboard")

# ====================== DATA LOADING ======================
@st.cache_data
def load_data():
    return pd.read_csv("final_with_predictions.csv")

df = load_data()
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(["user_id", "datetime"])

# Sidebar
st.sidebar.header("Filters")
users = df["user_id"].unique()
selected_user = st.sidebar.selectbox("Select User", users)

user_df = df[df["user_id"] == selected_user].copy().sort_values("datetime")
latest = user_df.iloc[-1]

# ====================== USER PROFILE ======================
gender = "Female" if str(latest.get("sex", "")).lower() == "female" else "Male" if str(latest.get("sex", "")).lower() == "male" else "N/A"
initial = gender[0] if gender != "N/A" else "U"

st.markdown(f"""
<div class="user-card">
    <div style="display:flex; align-items:center;">
        <div class="initials">{initial}</div>
        <div>
            <h2 style="margin:0; color:white;">{selected_user}</h2>
            <p style="margin:8px 0 0 0; color:#94a3b8; font-size:18px;">
                {latest.get('age', 'N/A')} years • {gender} • {latest.get('Sport_type', 'N/A')}
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================== HEALTH INDICATORS ======================
st.subheader("Health Indicators")

col1, col2, col3 = st.columns(3)

def get_status_badge(metric_type):
    if metric_type == "body_age":
        delta = latest.get("body_age_change", 0)
        if delta <= -1: return "OPTIMAL", "#10b981"
        elif delta <= 1: return "GOOD", "#eab308"
        else: return "ATTENTION", "#ef4444"
    elif metric_type == "work_load":
        value = latest.get("work_load", 0)
        if value <= 30: return "OPTIMAL", "#10b981"
        elif value <= 60: return "MODERATE", "#eab308"
        else: return "HIGH", "#ef4444"
    else:  # body_toxin
        value = latest.get("body_toxin", 0)
        if value <= 1: return "OPTIMAL", "#10b981"
        elif value <= 3: return "MODERATE", "#eab308"
        else: return "HIGH", "#ef4444"

# Body Age
status, color = get_status_badge("body_age")
delta_body = latest.get("body_age_change", 0)
with col1:
    st.markdown(f"""
    <div class="health-card" style="border-color: {color};">
        <div style="display:flex; justify-content:space-between;">
            <span style="font-size:28px;">❤️</span>
            <span style="background:{color}; color:white; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:bold;">{status}</span>
        </div>
        <h3 style="color:#e0f2e9; margin:15px 0 8px 0;">Body Age</h3>
        <h1 style="margin:0; color:white;">{latest.get('body_age', 0):.1f} <span style="font-size:22px; color:#94a3b8;">/ {latest.get('age', 0)}</span></h1>
        <p style="color:#94a3b8;">Biological vs Chronological Age</p>
        <p style="color:{'#4ade80' if delta_body <= 0 else '#f87171'}; font-size:15px; margin-top:auto;">
            {'↓' if delta_body <= 0 else '↑'} {abs(delta_body):.1f} years vs last period
        </p>
    </div>
    """, unsafe_allow_html=True)

# Work Load
status, color = get_status_badge("work_load")
delta_work = latest.get("work_load_change", 0)
with col2:
    st.markdown(f"""
    <div class="health-card" style="border-color: {color};">
        <div style="display:flex; justify-content:space-between;">
            <span style="font-size:28px;">⚡</span>
            <span style="background:{color}; color:white; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:bold;">{status}</span>
        </div>
        <h3 style="color:#e0f2e9; margin:15px 0 8px 0;">Workload Intensity</h3>
        <h1 style="margin:0; color:white;">{latest.get('work_load', 0):.1f} <span style="font-size:18px; color:#94a3b8;">/ 100</span></h1>
        <p style="color:{'#4ade80' if delta_work <= 0 else '#f87171'}; font-size:15px; margin-top:auto;">
            {'↓' if delta_work <= 0 else '↑'} {abs(delta_work):.1f} vs last period
        </p>
    </div>
    """, unsafe_allow_html=True)

# Body Toxins
status, color = get_status_badge("body_toxin")
delta_toxin = latest.get("body_toxin_change", 0)
with col3:
    st.markdown(f"""
    <div class="health-card" style="border-color: {color};">
        <div style="display:flex; justify-content:space-between;">
            <span style="font-size:28px;">🛡️</span>
            <span style="background:{color}; color:white; padding:6px 14px; border-radius:20px; font-size:13px; font-weight:bold;">{status}</span>
        </div>
        <h3 style="color:#e0f2e9; margin:15px 0 8px 0;">Body Toxins</h3>
        <h1 style="margin:0; color:white;">{latest.get('body_toxin', 0):.1f} <span style="font-size:18px; color:#94a3b8;">/ 100</span></h1>
        <p style="color:{'#4ade80' if delta_toxin <= 0 else '#f87171'}; font-size:15px; margin-top:auto;">
            {'↓' if delta_toxin <= 0 else '↑'} {abs(delta_toxin):.1f} vs last period
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ====================== AI PREDICTIONS ======================
st.subheader("AI Predictions")
st.caption("Predictions for the coming period based on your current profile and trends")

p1, p2, p3, p4 = st.columns(4)

# 1. Injury Risk
injury_risk = latest.get('predicted_injury_risk_%', 0)
if injury_risk < 25:
    injury_color = "#10b981"
    injury_text = "Low risk – Good recovery and load management"
elif injury_risk < 50:
    injury_color = "#f59e0b"
    injury_text = "Moderate risk – Pay attention to recovery"
else:
    injury_color = "#ef4444"
    injury_text = "High risk – Consider reducing intensity"

with p1:
    st.markdown(f"""
    <div style="background:#1a1a1a; padding:20px; border-radius:16px; border:1px solid {injury_color}; height:240px;">
        <h4 style="margin:0 0 8px 0; color:#e0f2e9;">Injury Risk</h4>
        <h1 style="margin:0; color:{injury_color};">{injury_risk:.0f}%</h1>
        <p style="color:#94a3b8; font-size:14px; margin:12px 0;">Probability of injury in the near future</p>
        <p style="color:{injury_color}; font-size:13px;">{injury_text}</p>
    </div>
    """, unsafe_allow_html=True)

# 2. Chronic Disease Risk
chronic_risk = latest.get('predicted_chronic_risk_%', 0)
if chronic_risk < 30:
    chronic_color = "#10b981"
    chronic_text = "Low risk – Current lifestyle is protective"
elif chronic_risk < 60:
    chronic_color = "#f59e0b"
    chronic_text = "Moderate risk – Some improvements recommended"
else:
    chronic_color = "#ef4444"
    chronic_text = "High risk – Medical follow-up advised"

with p2:
    st.markdown(f"""
    <div style="background:#1a1a1a; padding:20px; border-radius:16px; border:1px solid {chronic_color}; height:240px;">
        <h4 style="margin:0 0 8px 0; color:#e0f2e9;">Chronic Disease Risk</h4>
        <h1 style="margin:0; color:{chronic_color};">{chronic_risk:.0f}%</h1>
        <p style="color:#94a3b8; font-size:14px; margin:12px 0;">Risk of developing long-term health issues</p>
        <p style="color:{chronic_color}; font-size:13px;">{chronic_text}</p>
    </div>
    """, unsafe_allow_html=True)

# 3. Predicted Body Age
pred_body_age = latest.get('predicted_body_age_3m', 0)
delta_age = pred_body_age - latest.get('body_age', 0)
if delta_age < -1:
    age_color = "#10b981"
    age_text = "Excellent – Your body is getting biologically younger"
elif delta_age < 1:
    age_color = "#eab308"
    age_text = "Stable – Maintaining good biological age"
else:
    age_color = "#ef4444"
    age_text = "Attention – Biological age is increasing"

with p3:
    st.markdown(f"""
    <div style="background:#1a1a1a; padding:20px; border-radius:16px; border:1px solid {age_color}; height:240px;">
        <h4 style="margin:0 0 8px 0; color:#e0f2e9;">Predicted Body Age</h4>
        <h1 style="margin:0; color:{age_color};">{pred_body_age:.1f}</h1>
        <p style="color:#94a3b8; font-size:14px; margin:12px 0;">Estimated biological age in the near future</p>
        <p style="color:{age_color}; font-size:13px;">{age_text}</p>
    </div>
    """, unsafe_allow_html=True)

# 4. Performance Improvement
perf_imp = latest.get('predicted_performance_improvement_%', 0)
if perf_imp > 15:
    perf_color = "#10b981"
    perf_text = "Strong expected improvement"
elif perf_imp > 0:
    perf_color = "#eab308"
    perf_text = "Moderate expected improvement"
else:
    perf_color = "#ef4444"
    perf_text = "Risk of performance decline"

with p4:
    st.markdown(f"""
    <div style="background:#1a1a1a; padding:20px; border-radius:16px; border:1px solid {perf_color}; height:240px;">
        <h4 style="margin:0 0 8px 0; color:#e0f2e9;">Performance Improvement</h4>
        <h1 style="margin:0; color:{perf_color};">{perf_imp:+.1f}%</h1>
        <p style="color:#94a3b8; font-size:14px; margin:12px 0;">Expected change in physical performance</p>
        <p style="color:{perf_color}; font-size:13px;">{perf_text}</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ====================== LONGITUDINAL TIMELINE ======================
st.subheader("Longitudinal Performance Timeline")
st.caption("Monthly tracking with event correlation")

# 4 Boutons tous en noir
col_buttons = st.columns([1.2, 1, 1, 1])

btn_all = col_buttons[0].button("All Metrics", use_container_width=True)
btn_body = col_buttons[1].button("Body Age", use_container_width=True)
btn_work = col_buttons[2].button("Workload", use_container_width=True)
btn_toxin = col_buttons[3].button("Toxins", use_container_width=True)

# Logique des boutons
if btn_body:
    metrics = ["body_age"]
    colors = {"body_age": "#FFD700"}
elif btn_work:
    metrics = ["work_load"]
    colors = {"work_load": "#00FF9D"}
elif btn_toxin:
    metrics = ["body_toxin"]
    colors = {"body_toxin": "#00CCFF"}
else:
    metrics = ["body_age", "work_load", "body_toxin"]
    colors = {"body_age": "#FFD700", "work_load": "#00FF9D", "body_toxin": "#00CCFF"}

fig = px.line(
    user_df,
    x="datetime",
    y=metrics,
    color_discrete_map=colors,
    markers=True
)

fig.update_layout(
    template="plotly_dark",
    height=520,
    plot_bgcolor="#1a1a1a",
    paper_bgcolor="#0a0a0a",
    xaxis_title="",
    yaxis_title="",
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    font=dict(color="#e0f2e9")
)

st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Assessment History")
st.dataframe(user_df)
