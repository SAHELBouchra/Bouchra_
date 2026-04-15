import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide", page_title="PEREN AI – Digital Twin")

# ====================== STYLE CSS ======================
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
        margin-bottom: 30px;
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
    .health-card {
        background: #1a1a1a;
        padding: 22px;
        border-radius: 16px;
        border: 1px solid;
        height: 260px !important;
        display: flex;
        flex-direction: column;
    }
    /* Boutons Timeline en noir */
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
    .insight-card {
        background: #1a1a1a;
        padding: 18px;
        border-radius: 12px;
        border-left: 4px solid #10b981;
    }
</style>
""", unsafe_allow_html=True)

st.title("PEREN AI – Digital Twin Dashboard")

# ====================== DATA ======================
@st.cache_data
def load_data():
    return pd.read_csv("final_with_predictions.csv")

df = load_data()
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values(["user_id", "datetime"])

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
    else:
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
            {'↓' if delta_body <= 0 else '↑'} {abs(delta_body):.1f} years
        </p>
    </div>
    """, unsafe_allow_html=True)

# Work Load & Body Toxins (identique)
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
            {'↓' if delta_work <= 0 else '↑'} {abs(delta_work):.1f}
        </p>
    </div>
    """, unsafe_allow_html=True)

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
            {'↓' if delta_toxin <= 0 else '↑'} {abs(delta_toxin):.1f}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ====================== AI PREDICTIONS (Améliorée) ======================
st.subheader("AI Predictions")
st.caption("Predictions based on your current trends and lifestyle data")

p1, p2, p3, p4 = st.columns(4)

# Injury Risk
injury = latest.get('predicted_injury_risk_%', 0)
if injury < 25:
    inj_color, inj_text = "#10b981", "Low risk – Keep current balance"
elif injury < 50:
    inj_color, inj_text = "#f59e0b", "Moderate risk – Focus on recovery"
else:
    inj_color, inj_text = "#ef4444", "High risk – Reduce load and improve sleep"

with p1:
    st.markdown(f"""
    <div style="background:#1a1a1a; padding:20px; border-radius:16px; border:1px solid {inj_color}; height:245px;">
        <span style="font-size:28px;">⚠️</span>
        <h4 style="margin:12px 0 8px 0;">Injury Risk</h4>
        <h1 style="margin:0; color:{inj_color};">{injury:.0f}%</h1>
        <p style="color:#94a3b8; font-size:13.5px;">Probability of injury</p>
        <p style="color:{inj_color}; font-size:13px; margin-top:8px;">{inj_text}</p>
    </div>
    """, unsafe_allow_html=True)

# Chronic Disease Risk
chronic = latest.get('predicted_chronic_risk_%', 0)
if chronic < 30:
    chr_color, chr_text = "#10b981", "Low risk – Protective lifestyle"
elif chronic < 60:
    chr_color, chr_text = "#f59e0b", "Moderate risk – Improvements needed"
else:
    chr_color, chr_text = "#ef4444", "High risk – Medical follow-up advised"

with p2:
    st.markdown(f"""
    <div style="background:#1a1a1a; padding:20px; border-radius:16px; border:1px solid {chr_color}; height:245px;">
        <span style="font-size:28px;">🩺</span>
        <h4 style="margin:12px 0 8px 0;">Chronic Disease Risk</h4>
        <h1 style="margin:0; color:{chr_color};">{chronic:.0f}%</h1>
        <p style="color:#94a3b8; font-size:13.5px;">Long-term health risk</p>
        <p style="color:{chr_color}; font-size:13px; margin-top:8px;">{chr_text}</p>
    </div>
    """, unsafe_allow_html=True)

# Predicted Body Age
pred_age = latest.get('predicted_body_age_3m', 0)
age_delta = pred_age - latest.get('body_age', 0)
if age_delta < -0.5:
    age_color, age_text = "#10b981", "Improving – Body getting younger"
elif age_delta < 0.5:
    age_color, age_text = "#eab308", "Stable biological age"
else:
    age_color, age_text = "#ef4444", "Increasing – Focus on recovery"

with p3:
    st.markdown(f"""
    <div style="background:#1a1a1a; padding:20px; border-radius:16px; border:1px solid {age_color}; height:245px;">
        <span style="font-size:28px;">🧬</span>
        <h4 style="margin:12px 0 8px 0;">Predicted Body Age</h4>
        <h1 style="margin:0; color:{age_color};">{pred_age:.1f}</h1>
        <p style="color:#94a3b8; font-size:13.5px;">Future biological age</p>
        <p style="color:{age_color}; font-size:13px; margin-top:8px;">{age_text}</p>
    </div>
    """, unsafe_allow_html=True)

# Performance Improvement
perf = latest.get('predicted_performance_improvement_%', 0)
if perf > 10:
    perf_color, perf_text = "#10b981", "Strong expected gains"
elif perf > 0:
    perf_color, perf_text = "#eab308", "Moderate improvement expected"
else:
    perf_color, perf_text = "#ef4444", "Risk of decline – Optimize recovery"

with p4:
    st.markdown(f"""
    <div style="background:#1a1a1a; padding:20px; border-radius:16px; border:1px solid {perf_color}; height:245px;">
        <span style="font-size:28px;">📈</span>
        <h4 style="margin:12px 0 8px 0;">Performance Improvement</h4>
        <h1 style="margin:0; color:{perf_color};">{perf:+.1f}%</h1>
        <p style="color:#94a3b8; font-size:13.5px;">Expected performance change</p>
        <p style="color:{perf_color}; font-size:13px; margin-top:8px;">{perf_text}</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ====================== KEY INSIGHTS & RECOMMENDATIONS ======================
st.subheader("Key Insights & Recommendations")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.markdown(f"""
    <div class="insight-card">
        <h4>📌 Main Insight</h4>
        <p>Your body age is currently <strong>{latest.get('body_age', 0):.1f}</strong> compared to your chronological age of <strong>{latest.get('age', 0)}</strong>.</p>
        <p>Workload is in <strong>{latest.get('workload_state', 'moderate_load').replace('_', ' ').title()}</strong> zone.</p>
    </div>
    """, unsafe_allow_html=True)

with insight_col2:
    st.markdown(f"""
    <div class="insight-card">
        <h4>💡 Top Recommendation</h4>
        <p>Focus on improving sleep and reducing sedentary time to lower chronic risk and support performance gains.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ====================== TIMELINE ======================
st.subheader("Longitudinal Performance Timeline")
st.caption("Monthly tracking with event correlation")

col_buttons = st.columns([1.2, 1, 1, 1])
btn_all = col_buttons[0].button("All Metrics", use_container_width=True)
btn_body = col_buttons[1].button("Body Age", use_container_width=True)
btn_work = col_buttons[2].button("Workload", use_container_width=True)
btn_toxin = col_buttons[3].button("Toxins", use_container_width=True)

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

fig = px.line(user_df, x="datetime", y=metrics, color_discrete_map=colors, markers=True)
fig.update_layout(template="plotly_dark", height=520, plot_bgcolor="#1a1a1a", paper_bgcolor="#0a0a0a",
                  legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))

st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Assessment History")
st.dataframe(user_df)
