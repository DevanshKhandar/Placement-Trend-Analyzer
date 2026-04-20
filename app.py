"""
╔══════════════════════════════════════════════════════════════╗
║           PLACEMENT TREND ANALYZER — DASHBOARD              ║
║     Data Analytics Mini Project | Streamlit Dashboard       ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import (accuracy_score, r2_score, mean_absolute_error,
                             mean_squared_error, confusion_matrix, classification_report)
from scipy.stats import chi2_contingency, f_oneway
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Placement Trend Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# GLASSMORPHISM CSS — GLOSSY TRANSPARENT BLUE
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* === BASE BACKGROUND === */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0c4a6e 100%);
        background-size: 200% 200%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* === FLOATING GLOSSY ORBS EFFECT === */
    .stApp::before {
        content: '';
        position: fixed;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background:
            radial-gradient(circle at 20% 30%, rgba(56, 189, 248, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(99, 102, 241, 0.15) 0%, transparent 50%),
            radial-gradient(circle at 50% 50%, rgba(14, 165, 233, 0.1) 0%, transparent 60%);
        pointer-events: none;
        z-index: 0;
    }

    /* Override Streamlit Typography for Dark Mode */
    h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #f8fafc !important;
    }

    /* === GLASS CARD BASE (GLOSSY NAVY/BLUE) === */
    .glass-card {
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border: 1px solid rgba(125, 211, 252, 0.2);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow:
            0 8px 32px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow:
            0 12px 40px rgba(14, 165, 233, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }

    /* === HERO HEADER === */
    .hero-header {
        background: rgba(30, 58, 138, 0.3);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(125, 211, 252, 0.3);
        border-radius: 24px;
        padding: 36px 40px;
        margin-bottom: 28px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7dd3fc, #38bdf8, #0284c7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #bae6fd !important;
        font-weight: 400;
        letter-spacing: 0.2px;
    }

    /* === KPI METRIC CARDS === */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 28px;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 18px;
        padding: 22px 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease;
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 36px rgba(14, 165, 233, 0.3);
    }
    .kpi-icon {
        font-size: 2.2rem;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #f0f9ff !important;
        text-shadow: 0 2px 10px rgba(56, 189, 248, 0.4);
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #93c5fd !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* === SIDEBAR === */
    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.5) !important;
        backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.2);
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #7dd3fc !important;
    }

    /* === STREAMLIT INPUTS === */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background-color: #0f172a !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        color: #f8fafc !important;
    }
    .stSelectbox > div > div > div,
    .stSelectbox [data-baseweb="select"] span { color: #f8fafc !important; }

    /* ══ DROPDOWN POPOVER — NUCLEAR DARK OVERRIDE ══ */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] > div > div,
    div[data-baseweb="popover"] > div > div > div {
        background-color: #0f172a !important;
        background: #0f172a !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
    }
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] > div {
        background-color: #0f172a !important;
        background: #0f172a !important;
    }
    ul[role="listbox"] {
        background-color: #0f172a !important;
        background: #0f172a !important;
    }
    ul[role="listbox"] li {
        background-color: #0f172a !important;
        background: #0f172a !important;
        color: #e2e8f0 !important;
    }
    ul[role="listbox"] li:hover,
    ul[role="listbox"] li[aria-selected="true"],
    ul[role="listbox"] li:focus {
        background-color: #1e3a8a !important;
        background: #1e3a8a !important;
        color: #7dd3fc !important;
    }
    /* Target EVERY child inside popover */
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li *,
    div[data-baseweb="popover"] span,
    div[data-baseweb="popover"] div {
        color: #e2e8f0 !important;
    }
    div[data-baseweb="popover"] li:hover,
    div[data-baseweb="popover"] li:hover * {
        background-color: #1e3a8a !important;
        background: #1e3a8a !important;
        color: #7dd3fc !important;
    }
    /* Override inline style="background-color: ..." on option items */
    [data-baseweb="menu"] [role="option"],
    [data-baseweb="menu"] [role="option"] * {
        background-color: #0f172a !important;
        background: #0f172a !important;
        color: #e2e8f0 !important;
    }
    [data-baseweb="menu"] [role="option"]:hover,
    [data-baseweb="menu"] [role="option"]:hover *,
    [data-baseweb="menu"] [role="option"][aria-selected="true"],
    [data-baseweb="menu"] [role="option"][aria-selected="true"] * {
        background-color: #1e3a8a !important;
        background: #1e3a8a !important;
        color: #7dd3fc !important;
    }
    /* SVG icons inside selectbox */
    .stSelectbox svg { fill: #38bdf8 !important; }

    /* Streamlit metric labels & values */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #f8fafc !important;
    }
    /* Dataframe / table styling */
    .stDataFrame, .stTable { color: #f8fafc !important; }

    /* Fix tabs text */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] p {
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 1rem;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 189, 248, 0.2) !important;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] p {
        color: #38bdf8 !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7, #3b82f6) !important;
        border: 1px solid #7dd3fc !important;
        color: white !important;
    }

    /* Tooltips / Metrics */
    .prediction-value { color: #f8fafc !important; }
    .prediction-label { color: #93c5fd !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOAD & PREPARE DATA (UG ENGINEERING FOCUS)
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/Placement_Data.csv')

    # Rename basic columns
    df.rename(columns={
        'sl_no': 'serial_no',
        'ssc_p': 'ssc_percentage',
        'ssc_b': 'ssc_board',
        'hsc_p': 'hsc_percentage',
        'hsc_b': 'hsc_board',
        'hsc_s': 'hsc_stream',
        'etest_p': 'employability_test_percentage',
    }, inplace=True)

    # Convert to pure Engineering UG Dataset Context
    df_clean = df.copy()

    # 1. Transform degree_type to Engineering Branches
    # Deterministic mapping so filtering remains stable
    np.random.seed(42)  
    engg_branches = ['Computer Engineering', 'Information Technology', 'Electronics & Telecomm.', 'Mechanical Engineering', 'Civil Engineering']
    # Add synthetic branch based on probability
    df_clean['engineering_branch'] = np.random.choice(engg_branches, size=len(df_clean), p=[0.35, 0.25, 0.20, 0.10, 0.10])
    
    # Remove old degree_t & degree_p (we will rename degree_p to cgpa percentage equivalent)
    df_clean['ug_percentage'] = df_clean['degree_p']
    df_clean.drop(columns=['degree_p', 'degree_t'], inplace=True)

    # 2. Transform MBA details to Core Technical Skills (branch-aligned)
    # Map each branch to realistic technical skill pools
    branch_skill_map = {
        'Computer Engineering':       ['Data Science & AI', 'Full-Stack Development', 'Cybersecurity'],
        'Information Technology':      ['Full-Stack Development', 'Data Science & AI', 'Cloud & DevOps'],
        'Electronics & Telecomm.':     ['Embedded Systems', 'VLSI Design', 'RF & Microwave Engg.'],
        'Mechanical Engineering':      ['CAD/CAM & Design', 'Mechatronics', 'Robotics & Automation'],
        'Civil Engineering':           ['Structural Analysis', 'CAD/CAM & Design', 'IoT & Smart Infra'],
    }
    # Assign a core_skill based on branch (deterministic via row index)
    def assign_skill(row):
        skills = branch_skill_map.get(row['engineering_branch'], ['General Engineering'])
        return skills[row.name % len(skills)]
    df_clean['core_skill'] = df_clean.apply(assign_skill, axis=1)
    df_clean.drop(columns=['specialisation'], inplace=True)

    # MBA Percentage -> Technical Interview Score
    df_clean['technical_interview_score'] = df_clean['mba_p']
    df_clean.drop(columns=['mba_p'], inplace=True)

    # 3. Drop Work Experience entirely as requested
    if 'workex' in df_clean.columns:
        df_clean.drop(columns=['workex'], inplace=True)

    # ═══ NEW: CGPA (derived from ug_percentage, scale 4-10) ═══
    df_clean['cgpa'] = np.round(df_clean['ug_percentage'] / 100 * 6 + 4, 2)

    # ═══ NEW: Internships (0-3, correlated with placement) ═══
    placed_mask = df_clean['status'] == 'Placed'
    df_clean.loc[placed_mask, 'internships'] = np.random.choice(
        [0,1,2,3], size=placed_mask.sum(), p=[.10,.25,.40,.25])
    df_clean.loc[~placed_mask, 'internships'] = np.random.choice(
        [0,1,2,3], size=(~placed_mask).sum(), p=[.40,.35,.20,.05])
    df_clean['internships'] = df_clean['internships'].astype(int)

    # ═══ NEW: Extracurricular Activities ═══
    acts = ['Sports','Coding Club','Cultural','Robotics','None','Hackathons','NSS/NCC']
    df_clean.loc[placed_mask, 'extracurricular'] = np.random.choice(
        acts, size=placed_mask.sum(), p=[.12,.25,.10,.15,.08,.20,.10])
    df_clean.loc[~placed_mask, 'extracurricular'] = np.random.choice(
        acts, size=(~placed_mask).sum(), p=[.15,.10,.15,.05,.30,.10,.15])

    # ═══ FEATURE ENGINEERING ═══
    df_clean['cgpa_category'] = pd.cut(df_clean['cgpa'], bins=[0,5.5,6.5,7.5,8.5,10],
        labels=['Below Avg','Average','Good','Very Good','Excellent'])
    df_clean['academic_consistency'] = df_clean[['ssc_percentage','hsc_percentage','ug_percentage']].std(axis=1).round(2)
    df_clean['overall_academic_score'] = (df_clean['ssc_percentage']*.2 + df_clean['hsc_percentage']*.3 + df_clean['ug_percentage']*.5).round(2)
    df_clean['has_extra'] = (df_clean['extracurricular'] != 'None').astype(int)
    df_clean['activity_score'] = df_clean['internships'] * 2 + df_clean['has_extra']
    df_clean['placement_readiness'] = (df_clean['overall_academic_score']*.4 + df_clean['employability_test_percentage']*.2 + df_clean['technical_interview_score']*.2 + df_clean['activity_score']*3).round(2)
    df_clean['cgpa_x_internships'] = (df_clean['cgpa'] * df_clean['internships']).round(2)
    df_clean['cgpa_squared'] = (df_clean['cgpa'] ** 2).round(2)

    # Fill missing salary
    df_clean['salary'].fillna(0, inplace=True)
    df_clean.drop_duplicates(inplace=True)

    return df_clean

df = load_data()
placed_df = df[df['status'] == 'Placed']
not_placed_df = df[df['status'] == 'Not Placed']


# ──────────────────────────────────────────────
# TRAIN ML MODELS (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def train_models(dataframe):
    df_ml = dataframe.copy()
    encoders = {}
    for col in ['gender','core_skill','engineering_branch','hsc_stream','status','extracurricular','cgpa_category']:
        le = LabelEncoder()
        df_ml[f'{col}_enc'] = le.fit_transform(df_ml[col].astype(str))
        encoders[col] = le

    features = ['ssc_percentage','hsc_percentage','ug_percentage','employability_test_percentage',
                'technical_interview_score','cgpa','internships',
                'gender_enc','core_skill_enc','engineering_branch_enc','hsc_stream_enc',
                'overall_academic_score','academic_consistency','activity_score',
                'placement_readiness','has_extra','cgpa_x_internships']

    # ── 1. Logistic Regression (Classification) ──
    X_c = df_ml[features]; y_c = df_ml['status_enc']
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_c, y_c, test_size=.2, random_state=42)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(Xtr_c, ytr_c)
    clf_pred = clf.predict(Xte_c)
    clf_acc = accuracy_score(yte_c, clf_pred)
    clf_cm = confusion_matrix(yte_c, clf_pred)
    clf_rpt = classification_report(yte_c, clf_pred, target_names=encoders['status'].classes_, output_dict=True)

    # ── 2. Linear Regression (Salary) ──
    pm = df_ml[df_ml['status']=='Placed']
    X_r = pm[features]; y_r = pm['salary']
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_r, y_r, test_size=.2, random_state=42)
    reg = LinearRegression()
    reg.fit(Xtr_r, ytr_r)
    reg_pred = reg.predict(Xte_r)
    reg_r2 = r2_score(yte_r, reg_pred)
    reg_mae = mean_absolute_error(yte_r, reg_pred)

    # ── 3. Polynomial Regression (Salary, degree=2) ──
    pf = ['cgpa','internships','employability_test_percentage','technical_interview_score']
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xtr_poly = poly.fit_transform(Xtr_r[pf])
    Xte_poly = poly.transform(Xte_r[pf])
    preg = LinearRegression()
    preg.fit(Xtr_poly, ytr_r)
    poly_pred = preg.predict(Xte_poly)
    poly_r2 = r2_score(yte_r, poly_pred)
    poly_mae = mean_absolute_error(yte_r, poly_pred)

    return {
        'clf': clf, 'clf_acc': clf_acc, 'clf_cm': clf_cm, 'clf_rpt': clf_rpt,
        'reg': reg, 'reg_r2': reg_r2, 'reg_mae': reg_mae, 'reg_pred': reg_pred, 'reg_actual': yte_r.values,
        'preg': preg, 'poly': poly, 'poly_r2': poly_r2, 'poly_mae': poly_mae, 'poly_pred': poly_pred,
        'encoders': encoders, 'features': features, 'poly_feats': pf,
        'coef_lr': dict(zip(features, reg.coef_.round(2))),
        'coef_logistic': dict(zip(features, clf.coef_[0].round(4))),
    }

models = train_models(df)
encoders = models['encoders']
feature_cols = models['features']

# ── STATISTICAL TESTS ──
@st.cache_data
def run_stat_tests(dataframe):
    res = {}
    # ANOVA: CGPA across branches
    groups = [g['cgpa'].values for _, g in dataframe.groupby('engineering_branch')]
    f1, p1 = f_oneway(*groups)
    stats1 = dataframe.groupby('engineering_branch')['cgpa'].agg(['mean','std','count']).round(3)
    gm = dataframe['cgpa'].mean()
    ssb = sum(len(g)*(g.mean()-gm)**2 for _,g in dataframe.groupby('engineering_branch')['cgpa'])
    ssw = sum(((g-g.mean())**2).sum() for _,g in dataframe.groupby('engineering_branch')['cgpa'])
    k = dataframe['engineering_branch'].nunique(); n = len(dataframe)
    res['anova'] = {'title':'ANOVA: CGPA across Branches','f':round(f1,4),'p':round(p1,4),'sig':p1<0.05,
        'stats':stats1,'gm':round(gm,3),'SSB':round(ssb,3),'SSW':round(ssw,3),
        'MSB':round(ssb/(k-1),3),'MSW':round(ssw/(n-k),3),'dfb':k-1,'dfw':n-k}
    # Chi-Square: Branch vs Placement
    ct = pd.crosstab(dataframe['engineering_branch'], dataframe['status'])
    chi2, p2, dof, exp = chi2_contingency(ct)
    res['chi2'] = {'title':'Chi²: Branch vs Placement','chi2':round(chi2,4),'p':round(p2,4),
        'dof':dof,'sig':p2<0.05,'observed':ct,
        'expected':pd.DataFrame(np.round(exp,2), index=ct.index, columns=ct.columns)}
    # Chi-Square: Internships vs Placement
    ct2 = pd.crosstab(dataframe['internships'], dataframe['status'])
    chi2b, p3, dof2, exp2 = chi2_contingency(ct2)
    res['chi2_intern'] = {'title':'Chi²: Internships vs Placement','chi2':round(chi2b,4),'p':round(p3,4),
        'dof':dof2,'sig':p3<0.05,'observed':ct2,
        'expected':pd.DataFrame(np.round(exp2,2), index=ct2.index, columns=ct2.columns)}
    return res

stat_results = run_stat_tests(df)


# ──────────────────────────────────────────────
# PLOTLY CHART THEME FOR DARK GLOSSY BLUE
# ──────────────────────────────────────────────
CHART_COLORS = ['#38bdf8', '#818cf8', '#34d399', '#f472b6', '#a78bfa', '#fbbf24']

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#f1f5f9'),
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(
        bgcolor='rgba(15, 23, 42, 0.9)',
        bordercolor='rgba(56, 189, 248, 0.5)',
        font=dict(family='Inter', size=13, color='#f8fafc')
    ),
    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title_font=dict(color='#cbd5e1'), tickfont=dict(color='#94a3b8')),
    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title_font=dict(color='#cbd5e1'), tickfont=dict(color='#94a3b8')),
)


# ──────────────────────────────────────────────
# SIDEBAR — FILTERS
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🎓 Student Filters")
    st.markdown("---")

    gender_filter = st.selectbox(
        "👤 Gender",
        options=['All'] + sorted(df['gender'].unique().tolist())
    )

    branch_filter = st.selectbox(
        "🛠️ Engineering Branch",
        options=['All'] + sorted(df['engineering_branch'].unique().tolist())
    )

    skill_filter = st.selectbox(
        "💻 Core Technical Skill",
        options=['All'] + sorted(df['core_skill'].unique().tolist())
    )

    stream_filter = st.selectbox(
        "📐 12th HSC Stream",
        options=['All'] + sorted(df['hsc_stream'].unique().tolist())
    )

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#bae6fd; font-size:0.75rem; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 10px;'>
        <strong>Dataset:</strong> Engineering UG Placements<br>
        <strong>Students:</strong> 215 records<br>
        <strong>Theme:</strong> Glassy Blue
    </div>
    """, unsafe_allow_html=True)


# Apply filters
filtered_df = df.copy()
if gender_filter != 'All':
    filtered_df = filtered_df[filtered_df['gender'] == gender_filter]
if branch_filter != 'All':
    filtered_df = filtered_df[filtered_df['engineering_branch'] == branch_filter]
if skill_filter != 'All':
    filtered_df = filtered_df[filtered_df['core_skill'] == skill_filter]
if stream_filter != 'All':
    filtered_df = filtered_df[filtered_df['hsc_stream'] == stream_filter]

filtered_placed = filtered_df[filtered_df['status'] == 'Placed']


# ──────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">📊 Placement Trend Analyzer</div>
    <div class="hero-subtitle">
        Engineering UG Placement Statistics & Predictive AI Dashboard
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# KPI METRICS
# ──────────────────────────────────────────────
total_students = len(filtered_df)
total_placed = len(filtered_placed)
placement_rate = round((total_placed / total_students * 100), 1) if total_students > 0 else 0
avg_salary = round(filtered_placed['salary'].mean()) if len(filtered_placed) > 0 else 0
max_salary = round(filtered_placed['salary'].max()) if len(filtered_placed) > 0 else 0

st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card">
        <div class="kpi-icon">👥</div>
        <div class="kpi-value">{total_students}</div>
        <div class="kpi-label">UG Students</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">✅</div>
        <div class="kpi-value">{placement_rate}%</div>
        <div class="kpi-label">Placement Rate</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">💰</div>
        <div class="kpi-value">₹{avg_salary:,.0f}</div>
        <div class="kpi-label">Average Salary</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-icon">🏆</div>
        <div class="kpi-value">₹{max_salary:,.0f}</div>
        <div class="kpi-label">Highest Package</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Overview",
    "💰 Package",
    "🛠️ Branch & Skills",
    "📊 Correlations",
    "🔬 Statistical Tests",
    "⚙️ Feature Engineering",
    "🤖 ML Models"
])


# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    # --- Placement Rate Pie ---
    with col1:
        status_counts = filtered_df['status'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.55,
            marker=dict(colors=['#34d399', '#f43f5e'], line=dict(color='rgba(255,255,255,0.1)', width=2)),
            textinfo='label+percent',
            textfont=dict(size=14, family='Inter', color='white')
        )])
        fig_pie.update_layout(
            **CHART_LAYOUT,
            title=dict(text='Overall Placement Rate', font=dict(size=18, color='#f1f5f9')),
            legend=dict(orientation='h', y=-0.1, x=0.5, xanchor='center', font=dict(color='#cbd5e1')),
            height=400,
            annotations=[dict(text=f'{placement_rate}%', x=0.5, y=0.5, font_size=32,
                             font_color='#7dd3fc', font_family='Inter',
                             showarrow=False)]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- Gender Distribution ---
    with col2:
        gender_status = filtered_df.groupby(['gender', 'status']).size().reset_index(name='count')
        fig_gender = px.bar(
            gender_status, x='gender', y='count', color='status',
            barmode='group',
            color_discrete_map={'Placed': '#34d399', 'Not Placed': '#f43f5e'},
            labels={'count': 'Students', 'gender': 'Gender', 'status': 'Status'}
        )
        fig_gender.update_layout(
            **CHART_LAYOUT,
            title=dict(text='Placement by Gender', font=dict(size=18, color='#f1f5f9')),
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
            height=400
        )
        fig_gender.update_xaxes(showgrid=False)
        fig_gender.update_traces(marker=dict(line=dict(width=0), cornerradius=8))
        st.plotly_chart(fig_gender, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 2 — SALARY ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    if len(filtered_placed) > 0:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=filtered_placed['salary'], nbinsx=20,
            marker=dict(color='rgba(56, 189, 248, 0.6)', line=dict(color='#38bdf8', width=2)),
            hovertemplate='Package Range: ₹%{x:,.0f}<br>Count: %{y}<extra></extra>'
        ))
        fig_hist.update_layout(**CHART_LAYOUT,
            title=dict(text='Salary Package Distribution', font=dict(size=18, color='#f1f5f9')),
            xaxis_title='Salary (₹)', yaxis_title='Students', height=400)
        fig_hist.update_xaxes(showgrid=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No placed students in current filter.")

    # Violin + Box plots
    v_c, b_c = st.columns(2)
    with v_c:
        if len(filtered_placed) > 0:
            fig_vio = px.violin(filtered_placed, x='engineering_branch', y='salary',
                color='engineering_branch', box=True, points='all',
                color_discrete_sequence=CHART_COLORS,
                labels={'salary':'Salary (₹)','engineering_branch':'Branch'})
            fig_vio.update_layout(**CHART_LAYOUT, title='Salary Violin Plot by Branch', height=420, showlegend=False)
            st.plotly_chart(fig_vio, use_container_width=True)
    with b_c:
        fig_bp = px.box(filtered_df, x='engineering_branch', y='cgpa', color='status',
            color_discrete_map={'Placed':'#34d399','Not Placed':'#f43f5e'},
            labels={'cgpa':'CGPA','engineering_branch':'Branch'})
        fig_bp.update_layout(**CHART_LAYOUT, title='CGPA Box Plot: Placed vs Not Placed', height=420)
        st.plotly_chart(fig_bp, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — ENGINEERING BRANCH & SKILLS
# ══════════════════════════════════════════════
with tab3:
    col1, col2 = st.columns(2)

    # --- Branch-wise Placement ---
    with col1:
        branch_data = filtered_df.groupby(['engineering_branch', 'status']).size().reset_index(name='count')
        fig_branch = px.bar(
            branch_data, x='engineering_branch', y='count', color='status',
            barmode='group',
            color_discrete_map={'Placed': '#38bdf8', 'Not Placed': '#f472b6'},
            labels={'count': 'Number of Students', 'engineering_branch': 'Engineering Branch'}
        )
        fig_branch.update_layout(
            **CHART_LAYOUT,
            title=dict(text='Engineering Branch vs Placement Status', font=dict(size=16, color='#f1f5f9')),
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
            height=420
        )
        fig_branch.update_xaxes(showgrid=False)
        fig_branch.update_traces(marker=dict(line=dict(width=0), cornerradius=6))
        st.plotly_chart(fig_branch, use_container_width=True)

    # --- Skills Breakdown ---
    with col2:
        skill_data = filtered_df.groupby(['core_skill', 'status']).size().reset_index(name='count')
        fig_skill = px.sunburst(
            skill_data, path=['core_skill', 'status'], values='count',
            color='status',
            color_discrete_map={'Placed': '#10b981', 'Not Placed': '#ef4444'}
        )
        fig_skill.update_layout(
            **CHART_LAYOUT,
            title=dict(text='Core Technical Skill Impact on Placements', font=dict(size=16, color='#f1f5f9')),
            height=420,
        )
        st.plotly_chart(fig_skill, use_container_width=True)

    # Row 2: Treemap + Extracurricular + Internships
    t_c, e_c = st.columns(2)
    with t_c:
        tree_data = filtered_df.groupby(['engineering_branch','status']).size().reset_index(name='count')
        fig_tree = px.treemap(tree_data, path=['engineering_branch','status'], values='count',
            color='status', color_discrete_map={'Placed':'#34d399','Not Placed':'#f43f5e'})
        fig_tree.update_layout(**CHART_LAYOUT, title='Treemap: Branch → Placement Status', height=400)
        st.plotly_chart(fig_tree, use_container_width=True)
    with e_c:
        ext_data = filtered_df.groupby(['extracurricular','status']).size().reset_index(name='count')
        fig_ext = px.bar(ext_data, x='extracurricular', y='count', color='status', barmode='group',
            color_discrete_map={'Placed':'#38bdf8','Not Placed':'#f472b6'},
            labels={'extracurricular':'Activity','count':'Students'})
        fig_ext.update_layout(**CHART_LAYOUT, title='Extracurricular Activities vs Placement', height=400)
        fig_ext.update_traces(marker=dict(line=dict(width=0), cornerradius=6))
        st.plotly_chart(fig_ext, use_container_width=True)

    # Radar chart: Branch-wise avg metrics
    radar_df = filtered_df.groupby('engineering_branch').agg(
        CGPA=('cgpa','mean'), Internships=('internships','mean'),
        ETest=('employability_test_percentage','mean'),
        TechScore=('technical_interview_score','mean'),
        PlacementRate=('status', lambda x: (x=='Placed').mean()*10)
    ).round(2)
    fig_radar = go.Figure()
    for branch in radar_df.index:
        fig_radar.add_trace(go.Scatterpolar(r=radar_df.loc[branch].values,
            theta=radar_df.columns, fill='toself', name=branch, opacity=0.6))
    fig_radar.update_layout(**CHART_LAYOUT, title='Radar: Branch-wise Performance Comparison',
        height=450, polar=dict(bgcolor='rgba(0,0,0,0)',
        radialaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#94a3b8')),
        angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='#cbd5e1'))))
    st.plotly_chart(fig_radar, use_container_width=True)
with tab4:
    col1, col2 = st.columns([1, 1])

    with col1:
        numerical_cols = ['ssc_percentage', 'hsc_percentage', 'ug_percentage',
                          'employability_test_percentage', 'technical_interview_score', 'salary']
        corr = filtered_df[numerical_cols].corr()

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=['10th %', '12th %', 'UG CGPA %', 'E-Test %', 'Tech Score %', 'Salary'],
            y=['10th %', '12th %', 'UG CGPA %', 'E-Test %', 'Tech Score %', 'Salary'],
            colorscale='Blues',
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate='%{text:.2f}',
            textfont=dict(size=13, color='white')
        ))
        fig_heatmap.update_layout(
            **CHART_LAYOUT,
            title=dict(text='Academic Scores vs Salary Heatmap', font=dict(size=16, color='#f1f5f9')),
            height=450
        )
        fig_heatmap.update_xaxes(side='bottom')
        st.plotly_chart(fig_heatmap, use_container_width=True)

    with col2:
        fig_scatter = px.scatter(
            filtered_df, x='technical_interview_score', y='salary',
            color='status', size='ug_percentage',
            color_discrete_map={'Placed': '#38bdf8', 'Not Placed': '#f472b6'},
            labels={'technical_interview_score': 'Tech Interview Score %', 'salary': 'Package (₹)'},
            opacity=0.8,
            hover_data=['engineering_branch']
        )
        fig_scatter.update_layout(
            **CHART_LAYOUT,
            title=dict(text='Tech Interview vs Salary Package', font=dict(size=16, color='#f1f5f9')),
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
            height=450
        )
        st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — STATISTICAL TESTS (ANOVA + Chi²)
# ══════════════════════════════════════════════
with tab5:
    st.markdown("### 🔬 Statistical Hypothesis Testing")

    # ── ANOVA ──
    a = stat_results['anova']
    st.markdown(f"#### 📐 {a['title']}")
    st.markdown(f"""
**Hypotheses:**
- **H₀:** Mean CGPA is equal across all engineering branches
- **H₁:** At least one branch has a different mean CGPA
""")
    c1, c2, c3 = st.columns(3)
    c1.metric("F-Statistic", f"{a['f']}")
    c2.metric("P-Value", f"{a['p']}")
    c3.metric("Result", "Reject H₀ ✅" if a['sig'] else "Fail to Reject H₀ ❌")

    st.markdown("**ANOVA Calculation Steps:**")
    calc_df = pd.DataFrame({
        'Grand Mean': [a['gm']], 'SSB (Between)': [a['SSB']], 'SSW (Within)': [a['SSW']],
        'MSB': [a['MSB']], 'MSW': [a['MSW']], 'df (Between)': [a['dfb']], 'df (Within)': [a['dfw']],
        'F = MSB/MSW': [a['f']]
    })
    st.dataframe(calc_df, use_container_width=True)
    st.markdown("**Branch-wise CGPA Statistics:**")
    st.dataframe(a['stats'], use_container_width=True)

    # Box plot for ANOVA
    fig_box = px.box(filtered_df, x='engineering_branch', y='cgpa', color='engineering_branch',
                     color_discrete_sequence=CHART_COLORS, labels={'cgpa':'CGPA','engineering_branch':'Branch'})
    fig_box.update_layout(**CHART_LAYOUT, title='CGPA Distribution by Branch', height=400, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # ── CHI-SQUARE ──
    ch = stat_results['chi2']
    st.markdown(f"#### 📊 {ch['title']}")
    st.markdown(f"""
**Hypotheses:**
- **H₀:** Engineering branch and placement status are independent
- **H₁:** There is a significant association between branch and placement
""")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("χ² Statistic", f"{ch['chi2']}")
    c2.metric("P-Value", f"{ch['p']}")
    c3.metric("Degrees of Freedom", f"{ch['dof']}")
    c4.metric("Result", "Reject H₀ ✅" if ch['sig'] else "Fail to Reject H₀ ❌")

    o_c, e_c = st.columns(2)
    with o_c:
        st.markdown("**Observed Frequencies:**")
        st.dataframe(ch['observed'], use_container_width=True)
    with e_c:
        st.markdown("**Expected Frequencies:**")
        st.dataframe(ch['expected'], use_container_width=True)

    # ── CHI-SQUARE: Internships ──
    ci = stat_results['chi2_intern']
    st.markdown(f"#### 🏢 {ci['title']}")
    st.markdown("**H₀:** Internship count and placement are independent | **H₁:** They are associated")
    c1, c2, c3 = st.columns(3)
    c1.metric("χ² Statistic", f"{ci['chi2']}")
    c2.metric("P-Value", f"{ci['p']}")
    c3.metric("Result", "Reject H₀ ✅" if ci['sig'] else "Fail to Reject H₀ ❌")

    fig_intern = px.bar(filtered_df.groupby(['internships','status']).size().reset_index(name='count'),
        x='internships', y='count', color='status', barmode='group',
        color_discrete_map={'Placed':'#34d399','Not Placed':'#f43f5e'},
        labels={'internships':'Number of Internships','count':'Students'})
    fig_intern.update_layout(**CHART_LAYOUT, title='Internships vs Placement Status', height=350)
    st.plotly_chart(fig_intern, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 6 — FEATURE ENGINEERING
# ══════════════════════════════════════════════
with tab6:
    st.markdown("### ⚙️ Feature Engineering Pipeline")

    st.markdown("#### 🔧 Engineered Features & Formulas")
    fe_info = pd.DataFrame({
        'Feature': ['CGPA', 'Academic Consistency', 'Overall Academic Score', 'Activity Score',
                     'Placement Readiness Index', 'CGPA × Internships', 'CGPA²'],
        'Formula': [
            'ug_percentage / 100 × 6 + 4 (scale 4-10)',
            'std(SSC%, HSC%, UG%)',
            '0.2×SSC% + 0.3×HSC% + 0.5×UG%',
            'internships × 2 + has_extracurricular',
            '0.4×academic + 0.2×etest + 0.2×tech_score + 3×activity',
            'CGPA × internship_count (interaction)',
            'CGPA² (polynomial term)'
        ],
        'Type': ['Transformation', 'Aggregation', 'Weighted Avg', 'Composite', 'Composite Index', 'Interaction', 'Polynomial']
    })
    st.dataframe(fe_info, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📊 Feature Importance (Logistic Regression Coefficients)")
        coefs = models['coef_logistic']
        coef_df = pd.DataFrame({'Feature': list(coefs.keys()), 'Coefficient': list(coefs.values())})
        coef_df['Abs'] = coef_df['Coefficient'].abs()
        coef_df = coef_df.sort_values('Abs', ascending=True)
        fig_imp = px.bar(coef_df, x='Coefficient', y='Feature', orientation='h',
            color='Coefficient', color_continuous_scale='RdYlGn',
            labels={'Coefficient':'Log Reg Coefficient'})
        fig_imp.update_layout(**CHART_LAYOUT, title='Feature Impact on Placement', height=500)
        st.plotly_chart(fig_imp, use_container_width=True)

    with c2:
        st.markdown("#### 📈 Engineered Feature Correlation with Placement")
        eng_cols = ['cgpa','internships','overall_academic_score','academic_consistency',
                    'activity_score','placement_readiness','cgpa_x_internships']
        df_temp = filtered_df.copy()
        df_temp['placed_binary'] = (df_temp['status']=='Placed').astype(int)
        corr_vals = df_temp[eng_cols + ['placed_binary']].corr()['placed_binary'].drop('placed_binary')
        fig_corr = px.bar(x=corr_vals.values, y=corr_vals.index, orientation='h',
            color=corr_vals.values, color_continuous_scale='Viridis',
            labels={'x':'Correlation with Placement','y':'Feature'})
        fig_corr.update_layout(**CHART_LAYOUT, title='Feature-Placement Correlation', height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Sample of engineered features
    st.markdown("#### 📋 Sample Data with Engineered Features")
    show_cols = ['engineering_branch','cgpa','internships','extracurricular','overall_academic_score',
                 'academic_consistency','activity_score','placement_readiness','cgpa_x_internships','status']
    st.dataframe(filtered_df[show_cols].head(10), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 7 — ML MODELS (All 3 + Predictions)
# ══════════════════════════════════════════════
with tab7:
    st.markdown("### 🤖 Machine Learning Models — Results & Calculations")

    # Model comparison cards
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div style="color:#93c5fd; font-weight:600; text-transform:uppercase; letter-spacing:1px;">Model 1: Classification</div>
            <div style="font-size:1.6rem; font-weight:800; color:#38bdf8; margin:6px 0;">Logistic Regression</div>
            <div style="font-size:1.1rem; color:#34d399; font-weight:600;">Accuracy: {models['clf_acc']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div style="color:#93c5fd; font-weight:600; text-transform:uppercase; letter-spacing:1px;">Model 2: Regression</div>
            <div style="font-size:1.6rem; font-weight:800; color:#c084fc; margin:6px 0;">Linear Regression</div>
            <div style="font-size:1.1rem; color:#34d399; font-weight:600;">R² = {models['reg_r2']:.4f} | MAE = ₹{models['reg_mae']:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="glass-card" style="text-align:center;">
            <div style="color:#93c5fd; font-weight:600; text-transform:uppercase; letter-spacing:1px;">Model 3: Polynomial</div>
            <div style="font-size:1.6rem; font-weight:800; color:#f472b6; margin:6px 0;">Poly Regression (deg=2)</div>
            <div style="font-size:1.1rem; color:#34d399; font-weight:600;">R² = {models['poly_r2']:.4f} | MAE = ₹{models['poly_mae']:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    # Confusion Matrix + Classification Report
    cm_c, rpt_c = st.columns(2)
    with cm_c:
        st.markdown("#### Confusion Matrix (Logistic Regression)")
        cm = models['clf_cm']
        fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['Not Placed','Placed'], y=['Not Placed','Placed'],
            colorscale='Blues', text=cm, texttemplate='%{text}', textfont=dict(size=20, color='white')))
        fig_cm.update_layout(**CHART_LAYOUT, title='Predicted vs Actual', height=350,
            xaxis_title='Predicted', yaxis_title='Actual')
        st.plotly_chart(fig_cm, use_container_width=True)

    with rpt_c:
        st.markdown("#### Classification Report")
        rpt = models['clf_rpt']
        rpt_df = pd.DataFrame({
            'Class': ['Not Placed','Placed','Weighted Avg'],
            'Precision': [rpt.get('Not Placed',{}).get('precision',0), rpt.get('Placed',{}).get('precision',0), rpt.get('weighted avg',{}).get('precision',0)],
            'Recall': [rpt.get('Not Placed',{}).get('recall',0), rpt.get('Placed',{}).get('recall',0), rpt.get('weighted avg',{}).get('recall',0)],
            'F1-Score': [rpt.get('Not Placed',{}).get('f1-score',0), rpt.get('Placed',{}).get('f1-score',0), rpt.get('weighted avg',{}).get('f1-score',0)],
        })
        st.dataframe(rpt_df.round(3), use_container_width=True, hide_index=True)

        st.markdown("#### Linear Regression Coefficients (Top 5)")
        lr_coef = models['coef_lr']
        top5 = sorted(lr_coef.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        st.dataframe(pd.DataFrame(top5, columns=['Feature','Coefficient']), use_container_width=True, hide_index=True)

    # Actual vs Predicted plots
    act_c, poly_c = st.columns(2)
    with act_c:
        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(x=models['reg_actual'], y=models['reg_pred'], mode='markers',
            marker=dict(color='#38bdf8', size=8), name='Predictions'))
        mn = min(min(models['reg_actual']), min(models['reg_pred']))
        mx = max(max(models['reg_actual']), max(models['reg_pred']))
        fig_ap.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
            line=dict(color='#f43f5e', dash='dash'), name='Perfect Fit'))
        fig_ap.update_layout(**CHART_LAYOUT, title='Linear Reg: Actual vs Predicted Salary',
            xaxis_title='Actual (₹)', yaxis_title='Predicted (₹)', height=400)
        st.plotly_chart(fig_ap, use_container_width=True)

    with poly_c:
        fig_pp = go.Figure()
        fig_pp.add_trace(go.Scatter(x=models['reg_actual'], y=models['poly_pred'], mode='markers',
            marker=dict(color='#f472b6', size=8), name='Poly Predictions'))
        fig_pp.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode='lines',
            line=dict(color='#34d399', dash='dash'), name='Perfect Fit'))
        fig_pp.update_layout(**CHART_LAYOUT, title='Polynomial Reg: Actual vs Predicted Salary',
            xaxis_title='Actual (₹)', yaxis_title='Predicted (₹)', height=400)
        st.plotly_chart(fig_pp, use_container_width=True)

    # Model comparison bar chart
    st.markdown("#### 📊 Model Comparison Summary")
    comp_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Linear Regression', 'Polynomial Regression (deg=2)'],
        'Type': ['Classification', 'Regression', 'Regression'],
        'Primary Metric': [f"Accuracy: {models['clf_acc']*100:.1f}%", f"R² = {models['reg_r2']:.4f}", f"R² = {models['poly_r2']:.4f}"],
        'MAE': ['N/A', f"₹{models['reg_mae']:,.0f}", f"₹{models['poly_mae']:,.0f}"],
        'Features Used': [f"{len(models['features'])} engineered", f"{len(models['features'])} engineered", f"{len(models['poly_feats'])} key features (degree-2 poly)"],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Prediction Form
    st.markdown("#### 🔮 Predict Your Placement & Package")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        inp_ssc = st.slider("10th Percentage", 30.0, 100.0, 75.0, 0.5)
        inp_hsc = st.slider("12th Percentage", 30.0, 100.0, 80.0, 0.5)
        inp_gender = st.selectbox("Gender", ['M','F'], key='pred_gender')
    with col_b:
        inp_ug = st.slider("UG Percentage", 30.0, 100.0, 70.0, 0.5)
        inp_etest = st.slider("Employability Test %", 30.0, 100.0, 85.0, 0.5)
        inp_intern = st.slider("Internships (0-3)", 0, 3, 1)
    with col_c:
        inp_tech = st.slider("Tech Interview %", 30.0, 100.0, 75.0, 0.5)
        inp_skill = st.selectbox("Core Skill", sorted(df['core_skill'].unique()), key='pred_skill')
        inp_branch = st.selectbox("Branch", sorted(df['engineering_branch'].unique()), key='pred_branch')

    predict_btn = st.button("🔮 Run AI Prediction", type="primary", use_container_width=True)

    if predict_btn:
        try:
            g_e = encoders['gender'].transform([inp_gender])[0]
            s_e = encoders['core_skill'].transform([inp_skill])[0]
            b_e = encoders['engineering_branch'].transform([inp_branch])[0]
            h_e = encoders['hsc_stream'].transform(['Science'])[0]
            cgpa_v = inp_ug / 100 * 6 + 4
            oas = inp_ssc*.2 + inp_hsc*.3 + inp_ug*.5
            ac = np.std([inp_ssc, inp_hsc, inp_ug])
            act_s = inp_intern * 2 + 1
            pr = oas*.4 + inp_etest*.2 + inp_tech*.2 + act_s*3
            cxi = cgpa_v * inp_intern

            inp_arr = np.array([[inp_ssc, inp_hsc, inp_ug, inp_etest, inp_tech,
                                 cgpa_v, inp_intern, g_e, s_e, b_e, h_e,
                                 oas, ac, act_s, pr, 1, cxi]])

            pred = models['clf'].predict(inp_arr)[0]
            proba = models['clf'].predict_proba(inp_arr)[0]
            label = encoders['status'].inverse_transform([pred])[0]
            conf = max(proba) * 100
            sal = max(0, models['reg'].predict(inp_arr)[0])

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""<div class="glass-card" style="text-align:center;">
                    <div style="color:#93c5fd; font-weight:600;">Placement Prediction</div>
                    <div style="font-size:2.5rem; font-weight:800; color:{'#34d399' if label=='Placed' else '#f43f5e'};">
                        {'🎉' if label=='Placed' else '📚'} {label}</div>
                    <div style="color:#cbd5e1;">Confidence: <strong>{conf:.1f}%</strong></div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="glass-card" style="text-align:center;">
                    <div style="color:#93c5fd; font-weight:600;">Expected Package</div>
                    <div style="font-size:2.5rem; font-weight:800; color:#c084fc;">₹{sal:,.0f}</div>
                    <div style="color:#cbd5e1;">Linear Regression Prediction</div>
                </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.85rem; padding: 20px 0 10px; border-top: 1px solid rgba(56, 189, 248, 0.2); margin-top: 40px;">
    <strong>Placement Trend Analyzer</strong> — Engineering UG Dashboard | Built with ❤️ using Python, Streamlit & Plotly<br>
    <span style="color:#94a3b8;">Models: Logistic Regression · Linear Regression · Polynomial Regression | Tests: ANOVA · Chi-Square | Feature Engineering Applied</span>
</div>
""", unsafe_allow_html=True)
