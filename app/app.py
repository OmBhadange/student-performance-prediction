import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import time
import os
import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="AI Student Performance Predictor v2",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────

@st.cache_resource
def load_model():
    model_path = "models/student_score_predictor.pkl"
    if not os.path.exists(model_path):
        st.error("❌ Model file not found at `models/student_score_predictor.pkl`. Please train the model first.")
        st.stop()
    return joblib.load(model_path)

model = load_model()

# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────

for key, default in {
    "prediction_done": False,
    "score": 0.0,
    "grade": "",
    "active_tab": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def get_grade(score):
    if score >= 90: return "A+"
    elif score >= 80: return "A"
    elif score >= 70: return "B"
    elif score >= 60: return "C"
    else: return "D"

def get_percentile(score):
    """Approximate percentile based on normal distribution assumption."""
    mean, std = 67, 12
    z = (score - mean) / std
    from scipy.stats import norm
    return round(norm.cdf(z) * 100, 1)

def get_rank_label(percentile):
    if percentile >= 95: return "Top 5%", "#22C55E"
    elif percentile >= 90: return "Top 10%", "#3B82F6"
    elif percentile >= 75: return "Top 25%", "#8B5CF6"
    elif percentile >= 50: return "Top 50%", "#F59E0B"
    else: return "Below Average", "#EF4444"

def get_badge_html(score):
    if score >= 80:   return "<span class='badge badge-excellent'>🌟 Excellent</span>"
    elif score >= 60: return "<span class='badge badge-good'>👍 Good</span>"
    else:             return "<span class='badge badge-improve'>📚 Needs Work</span>"

def get_perf_color(score):
    if score >= 80: return "#22C55E"
    elif score >= 60: return "#3B82F6"
    else: return "#F59E0B"

LEVEL_MAP      = {"High": 0, "Low": 1, "Medium": 2}
YES_NO_MAP     = {"No": 0, "Yes": 1}
SCHOOL_MAP     = {"Private": 0, "Public": 1}
PEER_MAP       = {"Negative": 0, "Neutral": 1, "Positive": 2}
PARENT_EDU_MAP = {"College": 0, "High School": 1, "Postgraduate": 2}
DISTANCE_MAP   = {"Far": 0, "Moderate": 1, "Near": 2}
GENDER_MAP     = {"Female": 0, "Male": 1}

REQUIRED_COLS = [
    "Hours_Studied","Attendance","Parental_Involvement","Access_to_Resources",
    "Extracurricular_Activities","Sleep_Hours","Previous_Scores","Motivation_Level",
    "Internet_Access","Tutoring_Sessions","Family_Income","Teacher_Quality",
    "School_Type","Peer_Influence","Physical_Activity","Learning_Disabilities",
    "Parental_Education_Level","Distance_from_Home","Gender"
]

def encode_batch(df):
    """Encode categorical columns in a batch dataframe."""
    d = df.copy()
    for col in ["Parental_Involvement","Access_to_Resources","Motivation_Level","Family_Income","Teacher_Quality"]:
        if col in d.columns and d[col].dtype == object:
            d[col] = d[col].map(LEVEL_MAP)
    for col in ["Extracurricular_Activities","Internet_Access","Learning_Disabilities"]:
        if col in d.columns and d[col].dtype == object:
            d[col] = d[col].map(YES_NO_MAP)
    if "School_Type" in d.columns and d["School_Type"].dtype == object:
        d["School_Type"] = d["School_Type"].map(SCHOOL_MAP)
    if "Peer_Influence" in d.columns and d["Peer_Influence"].dtype == object:
        d["Peer_Influence"] = d["Peer_Influence"].map(PEER_MAP)
    if "Parental_Education_Level" in d.columns and d["Parental_Education_Level"].dtype == object:
        d["Parental_Education_Level"] = d["Parental_Education_Level"].map(PARENT_EDU_MAP)
    if "Distance_from_Home" in d.columns and d["Distance_from_Home"].dtype == object:
        d["Distance_from_Home"] = d["Distance_from_Home"].map(DISTANCE_MAP)
    if "Gender" in d.columns and d["Gender"].dtype == object:
        d["Gender"] = d["Gender"].map(GENDER_MAP)
    return d

def generate_pdf_report(name, score, grade, percentile, rank_label, input_dict, date_str):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=20*mm, leftMargin=20*mm,
                            topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle("Title", fontSize=22, alignment=TA_CENTER,
                                  textColor=colors.HexColor("#4F46E5"), spaceAfter=4,
                                  fontName="Helvetica-Bold")
    sub_style   = ParagraphStyle("Sub", fontSize=11, alignment=TA_CENTER,
                                  textColor=colors.HexColor("#64748B"), spaceAfter=16)
    story.append(Paragraph("🎓 AI Student Performance Report", title_style))
    story.append(Paragraph(f"Generated on {date_str}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 10))

    # Student name
    if name:
        name_style = ParagraphStyle("Name", fontSize=15, fontName="Helvetica-Bold",
                                     textColor=colors.HexColor("#1E293B"), spaceAfter=12)
        story.append(Paragraph(f"Student: {name}", name_style))

    # Score summary table
    grade_color = {"A+":"#22C55E","A":"#3B82F6","B":"#8B5CF6","C":"#F59E0B","D":"#EF4444"}.get(grade, "#64748B")
    summary_data = [
        ["Predicted Score", "Grade", "Percentile", "Rank"],
        [f"{score:.1f} / 100", grade, f"{percentile}th", rank_label],
    ]
    t = Table(summary_data, colWidths=[42*mm]*4)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#4F46E5")),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 11),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("ROWHEIGHT",   (0,0), (-1,-1), 18),
        ("FONTSIZE",    (0,1), (-1,1), 13),
        ("FONTNAME",    (0,1), (-1,1), "Helvetica-Bold"),
        ("TEXTCOLOR",   (1,1), (1,1), colors.HexColor(grade_color)),
        ("BACKGROUND",  (0,1), (-1,1), colors.HexColor("#F8FAFC")),
        ("BOX",         (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ("INNERGRID",   (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F8FAFC")]),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Academic factors
    section_style = ParagraphStyle("Section", fontSize=12, fontName="Helvetica-Bold",
                                    textColor=colors.HexColor("#4F46E5"), spaceBefore=10, spaceAfter=6)
    row_style = ParagraphStyle("Row", fontSize=10, textColor=colors.HexColor("#374151"))

    story.append(Paragraph("📚 Academic Factors", section_style))
    acad = [
        ["Hours Studied", str(input_dict["Hours_Studied"]), "Attendance", f"{input_dict['Attendance']}%"],
        ["Previous Score", str(input_dict["Previous_Scores"]), "Tutoring Sessions", str(input_dict["Tutoring_Sessions"])],
    ]
    at = Table(acad, colWidths=[45*mm, 30*mm, 45*mm, 30*mm])
    at.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",  (0,0), (-1,-1), 10),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#64748B")),
        ("TEXTCOLOR", (2,0), (2,-1), colors.HexColor("#64748B")),
        ("FONTNAME",  (1,0), (1,-1), "Helvetica-Bold"),
        ("FONTNAME",  (3,0), (3,-1), "Helvetica-Bold"),
        ("BOX",       (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ("BACKGROUND",(0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ("ROWHEIGHT", (0,0), (-1,-1), 16),
        ("LEFTPADDING",(0,0),(-1,-1), 8),
    ]))
    story.append(at)

    story.append(Paragraph("🧑 Personal Factors", section_style))
    pers = [
        ["Sleep Hours", str(input_dict["Sleep_Hours"]), "Physical Activity", f"{input_dict['Physical_Activity']} hrs/wk"],
        ["Gender", input_dict["_gender_raw"], "Extracurricular", input_dict["_extra_raw"]],
    ]
    pt = Table(pers, colWidths=[45*mm, 30*mm, 45*mm, 30*mm])
    pt.setStyle(TableStyle([
        ("FONTNAME",  (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE",  (0,0), (-1,-1), 10),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#64748B")),
        ("TEXTCOLOR", (2,0), (2,-1), colors.HexColor("#64748B")),
        ("FONTNAME",  (1,0), (1,-1), "Helvetica-Bold"),
        ("FONTNAME",  (3,0), (3,-1), "Helvetica-Bold"),
        ("BOX",       (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ("BACKGROUND",(0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ("ROWHEIGHT", (0,0), (-1,-1), 16),
        ("LEFTPADDING",(0,0),(-1,-1), 8),
    ]))
    story.append(pt)

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")))
    footer = ParagraphStyle("Footer", fontSize=8, alignment=TA_CENTER,
                             textColor=colors.HexColor("#94A3B8"), spaceBefore=8)
    story.append(Paragraph("AI Student Performance Predictor · Model: Linear Regression · Built by Om Avinash Bhadange", footer))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────

st.markdown("""
<style>
.stApp { background: #050816; color: #E2E8F0; }

section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.97);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
section[data-testid="stSidebar"] label { font-size: 13px !important; color: #94A3B8 !important; }

.hero-title {
    text-align: center; font-size: 48px; font-weight: 800;
    background: linear-gradient(90deg, #8B5CF6, #00D4FF, #22C55E);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.2; padding: 10px 0;
}
.hero-subtitle { text-align: center; color: #94A3B8; font-size: 16px; margin-bottom: 8px; }

.glass-card {
    background: rgba(255,255,255,0.04); border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.09); padding: 24px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25); margin-bottom: 16px;
}

.prediction-box {
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    border-radius: 22px; padding: 32px 20px; text-align: center; color: white;
    box-shadow: 0 0 40px rgba(99,102,241,0.4);
    animation: fadeSlideUp 0.4s ease;
}
.prediction-score { font-size: 64px; font-weight: 800; line-height: 1; }
.prediction-label { font-size: 14px; opacity: 0.8; margin-top: 6px; }
.grade-badge {
    display: inline-block; font-size: 32px; font-weight: 800;
    background: rgba(255,255,255,0.15); border-radius: 12px;
    padding: 6px 20px; margin-top: 10px; letter-spacing: 2px;
}

.rank-card {
    background: rgba(255,255,255,0.04); border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08); padding: 16px;
    text-align: center; margin-bottom: 10px;
}
.rank-val { font-size: 26px; font-weight: 700; }
.rank-lbl { font-size: 11px; color: #64748B; margin-top: 4px; }

.badge { display: inline-block; padding: 4px 12px; border-radius: 30px; font-size: 12px; font-weight: 600; margin-top: 8px; }
.badge-excellent { background: rgba(34,197,94,0.18); color: #4ADE80; border: 1px solid rgba(34,197,94,0.3); }
.badge-good      { background: rgba(59,130,246,0.18); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }
.badge-improve   { background: rgba(245,158,11,0.18); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); }

.tab-bar { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
.tab-btn {
    padding: 8px 18px; border-radius: 10px; font-size: 13px; font-weight: 500;
    border: 1px solid rgba(255,255,255,0.1); cursor: pointer;
    background: rgba(255,255,255,0.04); color: #CBD5E1;
    transition: all 0.2s;
}
.tab-btn.active { background: linear-gradient(90deg,#6366F1,#8B5CF6); color: white; border-color: transparent; }

.stButton > button {
    background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 10px 0 !important;
    font-size: 14px !important; font-weight: 600 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: scale(0.98) !important; }

.stDownloadButton > button {
    background: rgba(255,255,255,0.06) !important; color: #CBD5E1 !important;
    border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 10px !important;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04); border-radius: 14px;
    padding: 12px 16px; border: 1px solid rgba(255,255,255,0.07);
}
div[data-testid="metric-container"] label { color: #64748B !important; font-size: 12px !important; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #E2E8F0 !important; }

.batch-table th { background: rgba(99,102,241,0.25) !important; color: #E2E8F0 !important; }
.batch-table td { color: #CBD5E1 !important; }

@keyframes fadeSlideUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# PARTICLES
# ─────────────────────────────────────────

components.html("""
<div id="particles-js" style="position:fixed;width:100%;height:100%;top:0;left:0;z-index:-1;background:#050816;pointer-events:none;"></div>
<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
particlesJS('particles-js',{
  particles:{number:{value:60},color:{value:"#8B5CF6"},shape:{type:"circle"},
    opacity:{value:0.35,random:true},size:{value:2.2,random:true},
    line_linked:{enable:true,distance:140,color:"#00D4FF",opacity:0.2,width:1},
    move:{enable:true,speed:1,random:true,out_mode:"out"}},
  interactivity:{detect_on:"canvas",
    events:{onhover:{enable:true,mode:"grab"},onclick:{enable:true,mode:"push"}},
    modes:{grab:{distance:160,line_linked:{opacity:0.45}},push:{particles_nb:2}}}
});
</script>
""", height=0)

# ─────────────────────────────────────────
# SIDEBAR — INPUT PANEL
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📋 Student Information")
    st.markdown("---")
    student_name = st.text_input("Student Name (optional)", placeholder="e.g. Rahul Sharma")
    st.markdown("**📚 Academic Factors**")
    hours_studied    = st.slider("Hours Studied",       1,  44,  20)
    attendance       = st.slider("Attendance (%)",     50, 100,  80)
    previous_scores  = st.slider("Previous Score",     40, 100,  70)
    tutoring         = st.slider("Tutoring Sessions",   0,   8,   2)
    st.markdown("---")
    st.markdown("**🧑 Personal Factors**")
    sleep_hours       = st.slider("Sleep Hours",         4,  10,   7)
    physical_activity = st.slider("Physical Activity",   0,   6,   3)
    gender            = st.selectbox("Gender",          ["Female", "Male"])
    learning_dis      = st.selectbox("Learning Disabilities", ["No", "Yes"])
    extracurricular   = st.selectbox("Extracurricular", ["No", "Yes"])
    st.markdown("---")
    st.markdown("**🏡 Environment Factors**")
    parental_inv    = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
    access_res      = st.selectbox("Access to Resources",  ["Low", "Medium", "High"])
    motivation      = st.selectbox("Motivation Level",     ["Low", "Medium", "High"])
    internet        = st.selectbox("Internet Access",      ["No", "Yes"])
    family_income   = st.selectbox("Family Income",        ["Low", "Medium", "High"])
    teacher_quality = st.selectbox("Teacher Quality",      ["Low", "Medium", "High"])
    school_type     = st.selectbox("School Type",          ["Public", "Private"])
    peer_influence  = st.selectbox("Peer Influence",       ["Negative", "Neutral", "Positive"])
    parental_edu    = st.selectbox("Parental Education",   ["High School", "College", "Postgraduate"])
    distance_home   = st.selectbox("Distance from Home",   ["Near", "Moderate", "Far"])
    st.markdown("---")
    st.info("🤖 **Model:** Linear Regression\n\n📊 **R² ≈ 0.68** · v2.0\n\n🚀 Om Avinash Bhadange")

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────

st.markdown("<div class='hero-title'>🎓 AI Student Performance Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Predict · Grade · Rank · Batch Analyse · Export PDF</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LIVE METRICS
# ─────────────────────────────────────────

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("📚 Hours",    f"{hours_studied} h")
c2.metric("🏫 Attend",  f"{attendance}%")
c3.metric("📝 Prev",     previous_scores)
c4.metric("😴 Sleep",   f"{sleep_hours} h")
c5.metric("👨‍🏫 Tutor",   tutoring)
c6.metric("🏃 Activity", f"{physical_activity} h")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────

tab1, tab2 = st.tabs(["🎯 Single Prediction", "📂 Batch Prediction (CSV)"])

# ═══════════════════════════════════════
# TAB 1 — SINGLE PREDICTION
# ═══════════════════════════════════════

with tab1:
    btn_col, _ = st.columns([1, 2])
    with btn_col:
        predict_clicked = st.button("⚡ Predict Exam Score", use_container_width=True, key="single_pred")

    if predict_clicked:
        with st.spinner("Analysing student profile..."):
            time.sleep(0.35)

        input_df = pd.DataFrame([{
            "Hours_Studied": hours_studied, "Attendance": attendance,
            "Parental_Involvement": LEVEL_MAP[parental_inv],
            "Access_to_Resources": LEVEL_MAP[access_res],
            "Extracurricular_Activities": YES_NO_MAP[extracurricular],
            "Sleep_Hours": sleep_hours, "Previous_Scores": previous_scores,
            "Motivation_Level": LEVEL_MAP[motivation],
            "Internet_Access": YES_NO_MAP[internet],
            "Tutoring_Sessions": tutoring,
            "Family_Income": LEVEL_MAP[family_income],
            "Teacher_Quality": LEVEL_MAP[teacher_quality],
            "School_Type": SCHOOL_MAP[school_type],
            "Peer_Influence": PEER_MAP[peer_influence],
            "Physical_Activity": physical_activity,
            "Learning_Disabilities": YES_NO_MAP[learning_dis],
            "Parental_Education_Level": PARENT_EDU_MAP[parental_edu],
            "Distance_from_Home": DISTANCE_MAP[distance_home],
            "Gender": GENDER_MAP[gender],
        }])

        score = round(float(model.predict(input_df)[0]), 2)
        grade = get_grade(score)

        try:
            percentile = get_percentile(score)
        except ImportError:
            percentile = round(min(99, max(1, (score - 40) / 60 * 100)), 1)

        rank_label, rank_color = get_rank_label(percentile)
        st.session_state.update({"prediction_done": True, "score": score, "grade": grade})

        perf_color = get_perf_color(score)
        badge_html = get_badge_html(score)

        # ── Layout ──
        left, right = st.columns([1, 1], gap="large")

        with left:
            # Score + Grade card
            st.markdown(f"""
            <div class='glass-card'>
                <div class='prediction-box'>
                    {"<div style='font-size:13px;opacity:0.75;margin-bottom:4px'>" + student_name + "</div>" if student_name else ""}
                    <div class='prediction-score'>{score:.1f}</div>
                    <div class='prediction-label'>Predicted Exam Score / 100</div>
                    <div class='grade-badge'>Grade: {grade}</div>
                    <div style='margin-top:10px'>{badge_html}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Rank & Percentile cards
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f"<div class='rank-card'><div class='rank-val' style='color:{rank_color}'>{rank_label}</div><div class='rank-lbl'>Class Rank</div></div>", unsafe_allow_html=True)
            with r2:
                st.markdown(f"<div class='rank-card'><div class='rank-val' style='color:#8B5CF6'>{percentile}th</div><div class='rank-lbl'>Percentile</div></div>", unsafe_allow_html=True)
            with r3:
                st.markdown(f"<div class='rank-card'><div class='rank-val' style='color:{perf_color}'>{grade}</div><div class='rank-lbl'>Grade</div></div>", unsafe_allow_html=True)

            # Gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                title={"text": "Performance Gauge", "font": {"color": "#CBD5E1", "size": 13}},
                number={"font": {"color": "#E2E8F0", "size": 34}},
                gauge={
                    "axis": {"range": [0,100], "tickcolor": "#475569", "tickfont": {"color": "#64748B"}},
                    "bar": {"color": perf_color, "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(255,255,255,0.08)",
                    "steps": [
                        {"range": [0,60], "color": "rgba(239,68,68,0.12)"},
                        {"range": [60,80], "color": "rgba(245,158,11,0.12)"},
                        {"range": [80,100], "color": "rgba(34,197,94,0.12)"},
                    ],
                    "threshold": {"line": {"color": perf_color, "width": 3}, "thickness": 0.8, "value": score},
                }
            ))
            gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                font={"color":"#CBD5E1"}, height=240,
                                margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(gauge, use_container_width=True)
            if score >= 80: st.balloons()

        with right:
            # Factor bar chart
            factor_df = pd.DataFrame({
                "Factor": ["Hours Studied","Attendance","Previous Score","Sleep Hours","Tutoring","Activity"],
                "Value":  [hours_studied, attendance, previous_scores, sleep_hours, tutoring, physical_activity],
                "Max":    [44, 100, 100, 10, 8, 6],
            })
            factor_df["Pct"] = (factor_df["Value"] / factor_df["Max"] * 100).round(1)

            bar = px.bar(factor_df, x="Pct", y="Factor", orientation="h",
                         color="Pct", color_continuous_scale=["#6366F1","#00D4FF","#22C55E"],
                         text="Value", range_x=[0,115])
            bar.update_traces(textposition="outside", textfont_color="#CBD5E1")
            bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#CBD5E1", coloraxis_showscale=False,
                               xaxis=dict(showgrid=False, zeroline=False, tickfont_color="#64748B", title="Normalised %"),
                               yaxis=dict(showgrid=False, tickfont=dict(size=12, color="#CBD5E1")),
                               height=270, margin=dict(l=10,r=40,t=10,b=10))
            st.markdown("**📊 Key Performance Factors**")
            st.plotly_chart(bar, use_container_width=True)

            # Radar
            cats = ["Study","Attendance","Prev Score","Sleep","Activity"]
            vals = [hours_studied/44*100, attendance, previous_scores, sleep_hours/10*100, physical_activity/6*100]
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
                fillcolor="rgba(99,102,241,0.18)", line=dict(color="#8B5CF6", width=2)))
            radar.update_layout(
                polar=dict(bgcolor="rgba(0,0,0,0)",
                           radialaxis=dict(visible=True, range=[0,100], gridcolor="rgba(255,255,255,0.08)", tickfont_color="#475569"),
                           angularaxis=dict(gridcolor="rgba(255,255,255,0.08)", tickfont_color="#CBD5E1")),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=260,
                margin=dict(l=30,r=30,t=20,b=20))
            st.markdown("**🕸️ Student Profile Radar**")
            st.plotly_chart(radar, use_container_width=True)

        # ── PDF + TXT Download ──
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("**📄 Download Report**")

        input_dict_raw = {
            "Hours_Studied": hours_studied, "Attendance": attendance,
            "Previous_Scores": previous_scores, "Tutoring_Sessions": tutoring,
            "Sleep_Hours": sleep_hours, "Physical_Activity": physical_activity,
            "_gender_raw": gender, "_extra_raw": extracurricular,
        }
        date_str = datetime.now().strftime("%d %B %Y, %H:%M")
        pdf_buf = generate_pdf_report(student_name, score, grade, percentile, rank_label, input_dict_raw, date_str)

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.download_button("📥 Download PDF Report", data=pdf_buf,
                               file_name=f"student_report_{student_name or 'student'}.pdf",
                               mime="application/pdf", use_container_width=True)
        with dc2:
            txt = f"""AI STUDENT PERFORMANCE REPORT
Generated: {date_str}
Student: {student_name or 'N/A'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Predicted Score : {score:.2f} / 100
Grade           : {grade}
Percentile      : {percentile}th
Rank            : {rank_label}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hours Studied   : {hours_studied}
Attendance      : {attendance}%
Previous Score  : {previous_scores}
Sleep Hours     : {sleep_hours}
Tutoring        : {tutoring}
"""
            st.download_button("📄 Download TXT Report", data=txt,
                               file_name=f"student_report_{student_name or 'student'}.txt",
                               mime="text/plain", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif not st.session_state.prediction_done:
        st.markdown("""
        <div class='glass-card' style='text-align:center;padding:52px 20px;'>
            <div style='font-size:52px;margin-bottom:12px'>🎓</div>
            <div style='font-size:18px;color:#CBD5E1;font-weight:600'>Ready to predict</div>
            <div style='font-size:14px;color:#64748B;margin-top:8px'>
                Adjust the student factors in the sidebar, then click <b>⚡ Predict Exam Score</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ═══════════════════════════════════════

with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 📂 Batch Prediction — Upload CSV")
    st.markdown("""
    Upload a CSV file with student data. The file must contain a **Name** column plus the
    required feature columns. Categorical columns should use the same labels as the sidebar
    (e.g. `High`, `Medium`, `Low` for Parental_Involvement).
    """)

    # Sample CSV download
    sample = pd.DataFrame({
        "Name": ["Rahul Sharma","Priya Patel","Amit Singh","Sneha Rao","Arjun Mehta"],
        "Hours_Studied": [20,15,30,10,25],
        "Attendance": [85,78,92,65,88],
        "Previous_Scores": [75,70,88,55,80],
        "Sleep_Hours": [7,8,6,9,7],
        "Tutoring_Sessions": [2,1,3,0,2],
        "Physical_Activity": [3,4,2,5,3],
        "Parental_Involvement": ["High","Medium","High","Low","Medium"],
        "Access_to_Resources": ["High","Medium","High","Low","High"],
        "Extracurricular_Activities": ["Yes","No","Yes","No","Yes"],
        "Motivation_Level": ["High","Medium","High","Low","High"],
        "Internet_Access": ["Yes","Yes","Yes","No","Yes"],
        "Family_Income": ["High","Medium","High","Low","Medium"],
        "Teacher_Quality": ["High","Medium","High","Low","High"],
        "School_Type": ["Private","Public","Private","Public","Public"],
        "Peer_Influence": ["Positive","Neutral","Positive","Negative","Neutral"],
        "Learning_Disabilities": ["No","No","No","Yes","No"],
        "Parental_Education_Level": ["Postgraduate","College","Postgraduate","High School","College"],
        "Distance_from_Home": ["Near","Moderate","Near","Far","Near"],
        "Gender": ["Male","Female","Male","Female","Male"],
    })
    sample_csv = sample.to_csv(index=False).encode()
    st.download_button("⬇️ Download Sample CSV Template", data=sample_csv,
                       file_name="sample_students.csv", mime="text/csv")

    st.markdown("</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your student CSV", type=["csv"])

    if uploaded:
        with st.spinner("Reading file..."):
            df_raw = pd.read_csv(uploaded)

        st.markdown(f"**{len(df_raw)} students detected** · Previewing first 5 rows:")
        st.dataframe(df_raw.head(), use_container_width=True)

        # Validate columns
        missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {', '.join(missing_cols)}\n\nPlease use the sample template above.")
        else:
            if st.button("🚀 Run Batch Prediction", use_container_width=False, key="batch_pred"):
                with st.spinner(f"Predicting scores for {len(df_raw)} students..."):
                    time.sleep(0.3)
                    df_enc = encode_batch(df_raw[REQUIRED_COLS])
                    predictions = model.predict(df_enc)
                    df_raw["Predicted_Score"] = [round(float(p), 2) for p in predictions]
                    df_raw["Grade"]           = df_raw["Predicted_Score"].apply(get_grade)
                    try:
                        df_raw["Percentile"] = df_raw["Predicted_Score"].apply(get_percentile)
                    except ImportError:
                        df_raw["Percentile"] = df_raw["Predicted_Score"].apply(
                            lambda s: round(min(99, max(1, (s-40)/60*100)), 1))
                    df_raw["Performance"] = df_raw["Predicted_Score"].apply(
                        lambda s: "Excellent" if s >= 80 else ("Good" if s >= 60 else "Needs Work"))

                st.success(f"✅ Predictions complete for {len(df_raw)} students!")

                # Summary stats
                st.markdown("### 📊 Batch Summary")
                s1,s2,s3,s4 = st.columns(4)
                s1.metric("🏆 Highest Score",  f"{df_raw['Predicted_Score'].max():.1f}")
                s2.metric("📉 Lowest Score",   f"{df_raw['Predicted_Score'].min():.1f}")
                s3.metric("📊 Average Score",  f"{df_raw['Predicted_Score'].mean():.1f}")
                s4.metric("🎯 Pass Rate (≥60)", f"{(df_raw['Predicted_Score']>=60).mean()*100:.0f}%")

                # Grade distribution donut
                grade_counts = df_raw["Grade"].value_counts().reset_index()
                grade_counts.columns = ["Grade","Count"]
                grade_order = ["A+","A","B","C","D"]
                grade_colors = ["#22C55E","#3B82F6","#8B5CF6","#F59E0B","#EF4444"]
                grade_counts["Grade"] = pd.Categorical(grade_counts["Grade"], categories=grade_order, ordered=True)
                grade_counts = grade_counts.sort_values("Grade")

                bc1, bc2 = st.columns([1,2])
                with bc1:
                    donut = go.Figure(go.Pie(
                        labels=grade_counts["Grade"], values=grade_counts["Count"],
                        hole=0.55,
                        marker_colors=grade_colors[:len(grade_counts)],
                        textfont_color="white",
                    ))
                    donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#CBD5E1",
                                        showlegend=True, height=260,
                                        margin=dict(l=10,r=10,t=30,b=10),
                                        title=dict(text="Grade Distribution", font_color="#CBD5E1", x=0.5))
                    st.plotly_chart(donut, use_container_width=True)

                with bc2:
                    hist = px.histogram(df_raw, x="Predicted_Score", nbins=15,
                                        color_discrete_sequence=["#6366F1"],
                                        title="Score Distribution")
                    hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font_color="#CBD5E1", height=260,
                                       xaxis=dict(title="Score", gridcolor="rgba(255,255,255,0.06)"),
                                       yaxis=dict(title="Count", gridcolor="rgba(255,255,255,0.06)"),
                                       margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(hist, use_container_width=True)

                # Results table
                st.markdown("### 🏆 Student Rankings")
                name_col = "Name" if "Name" in df_raw.columns else df_raw.columns[0]
                results = df_raw[[name_col,"Predicted_Score","Grade","Percentile","Performance"]].copy()
                results = results.sort_values("Predicted_Score", ascending=False).reset_index(drop=True)
                results.index += 1
                results.index.name = "Rank"
                st.dataframe(
                    results.style
                    .background_gradient(subset=["Predicted_Score"], cmap="RdYlGn")
                    .format({"Predicted_Score": "{:.2f}", "Percentile": "{:.1f}th"}),
                    use_container_width=True
                )

                # Download results
                csv_out = results.to_csv().encode()
                st.download_button("📥 Download Results CSV", data=csv_out,
                                   file_name="batch_predictions.csv", mime="text/csv",
                                   use_container_width=False)

    else:
        st.markdown("""
        <div class='glass-card' style='text-align:center;padding:44px 20px;'>
            <div style='font-size:44px;margin-bottom:12px'>📂</div>
            <div style='font-size:17px;color:#CBD5E1;font-weight:600'>Upload a CSV file to predict scores for an entire class</div>
            <div style='font-size:13px;color:#64748B;margin-top:8px'>
                Download the sample template above, fill in your student data, and upload it here.
            </div>
        </div>
        """, unsafe_allow_html=True)