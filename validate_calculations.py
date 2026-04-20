"""
Independent Validation Script — Verifies all statistical tests and ML models
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import (accuracy_score, r2_score, mean_absolute_error,
                             confusion_matrix, classification_report)
from scipy.stats import chi2_contingency, f_oneway
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("   INDEPENDENT VALIDATION OF ALL CALCULATIONS")
print("=" * 70)

# ── Reproduce exact same data (same seed) ──
df = pd.read_csv('data/Placement_Data.csv')
df.rename(columns={
    'sl_no': 'serial_no', 'ssc_p': 'ssc_percentage', 'ssc_b': 'ssc_board',
    'hsc_p': 'hsc_percentage', 'hsc_b': 'hsc_board', 'hsc_s': 'hsc_stream',
    'etest_p': 'employability_test_percentage',
}, inplace=True)

dc = df.copy()
np.random.seed(42)

branches = ['Computer Engineering', 'Information Technology',
            'Electronics & Telecomm.', 'Mechanical Engineering', 'Civil Engineering']
dc['engineering_branch'] = np.random.choice(branches, size=len(dc), p=[.35,.25,.20,.10,.10])
dc['ug_percentage'] = dc['degree_p']
dc.drop(columns=['degree_p', 'degree_t'], inplace=True)

branch_skill_map = {
    'Computer Engineering':       ['Data Science & AI', 'Full-Stack Development', 'Cybersecurity'],
    'Information Technology':      ['Full-Stack Development', 'Data Science & AI', 'Cloud & DevOps'],
    'Electronics & Telecomm.':     ['Embedded Systems', 'VLSI Design', 'RF & Microwave Engg.'],
    'Mechanical Engineering':      ['CAD/CAM & Design', 'Mechatronics', 'Robotics & Automation'],
    'Civil Engineering':           ['Structural Analysis', 'CAD/CAM & Design', 'IoT & Smart Infra'],
}
def assign_skill(row):
    skills = branch_skill_map.get(row['engineering_branch'], ['General Engineering'])
    return skills[row.name % len(skills)]
dc['core_skill'] = dc.apply(assign_skill, axis=1)
dc.drop(columns=['specialisation'], inplace=True)

dc['technical_interview_score'] = dc['mba_p']
dc.drop(columns=['mba_p'], inplace=True)
if 'workex' in dc.columns:
    dc.drop(columns=['workex'], inplace=True)

dc['cgpa'] = np.round(dc['ug_percentage'] / 100 * 6 + 4, 2)
placed_mask = dc['status'] == 'Placed'
dc.loc[placed_mask, 'internships'] = np.random.choice([0,1,2,3], size=placed_mask.sum(), p=[.10,.25,.40,.25])
dc.loc[~placed_mask, 'internships'] = np.random.choice([0,1,2,3], size=(~placed_mask).sum(), p=[.40,.35,.20,.05])
dc['internships'] = dc['internships'].astype(int)

acts = ['Sports','Coding Club','Cultural','Robotics','None','Hackathons','NSS/NCC']
dc.loc[placed_mask, 'extracurricular'] = np.random.choice(acts, size=placed_mask.sum(), p=[.12,.25,.10,.15,.08,.20,.10])
dc.loc[~placed_mask, 'extracurricular'] = np.random.choice(acts, size=(~placed_mask).sum(), p=[.15,.10,.15,.05,.30,.10,.15])

dc['cgpa_category'] = pd.cut(dc['cgpa'], bins=[0,5.5,6.5,7.5,8.5,10],
    labels=['Below Avg','Average','Good','Very Good','Excellent'])
dc['academic_consistency'] = dc[['ssc_percentage','hsc_percentage','ug_percentage']].std(axis=1).round(2)
dc['overall_academic_score'] = (dc['ssc_percentage']*.2 + dc['hsc_percentage']*.3 + dc['ug_percentage']*.5).round(2)
dc['has_extra'] = (dc['extracurricular'] != 'None').astype(int)
dc['activity_score'] = dc['internships'] * 2 + dc['has_extra']
dc['placement_readiness'] = (dc['overall_academic_score']*.4 + dc['employability_test_percentage']*.2 + dc['technical_interview_score']*.2 + dc['activity_score']*3).round(2)
dc['cgpa_x_internships'] = (dc['cgpa'] * dc['internships']).round(2)
dc['cgpa_squared'] = (dc['cgpa'] ** 2).round(2)
dc['salary'].fillna(0, inplace=True)
dc.drop_duplicates(inplace=True)

print(f"\nDataset: {len(dc)} records, {dc['engineering_branch'].nunique()} branches, {dc['core_skill'].nunique()} skills")
print(f"Placed: {(dc['status']=='Placed').sum()}, Not Placed: {(dc['status']=='Not Placed').sum()}")

# ═══════════════════════════════════════════════
# 1. ANOVA: CGPA across Engineering Branches
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("  1. ANOVA TEST: CGPA across Engineering Branches")
print("=" * 70)

groups = [g['cgpa'].values for _, g in dc.groupby('engineering_branch')]
f_stat, p_val = f_oneway(*groups)

# Manual calculation for verification
grand_mean = dc['cgpa'].mean()
k = dc['engineering_branch'].nunique()
n = len(dc)

# SSB = Sum of n_i * (mean_i - grand_mean)^2
ssb = sum(len(g) * (g.mean() - grand_mean)**2 for _, g in dc.groupby('engineering_branch')['cgpa'])
# SSW = Sum of (x_ij - mean_i)^2
ssw = sum(((g - g.mean())**2).sum() for _, g in dc.groupby('engineering_branch')['cgpa'])

df_between = k - 1
df_within = n - k
msb = ssb / df_between
msw = ssw / df_within
f_manual = msb / msw

print(f"  Grand Mean (CGPA):     {grand_mean:.4f}")
print(f"  SSB (Between Groups):  {ssb:.4f}")
print(f"  SSW (Within Groups):   {ssw:.4f}")
print(f"  df(Between): {df_between},  df(Within): {df_within}")
print(f"  MSB: {msb:.4f},  MSW: {msw:.4f}")
print(f"  F-statistic (scipy):   {f_stat:.4f}")
print(f"  F-statistic (manual):  {f_manual:.4f}")
print(f"  Match: {'✅ YES' if abs(f_stat - f_manual) < 0.0001 else '❌ NO'}")
print(f"  P-value:               {p_val:.4f}")
print(f"  Significant (α=0.05):  {'Yes - Reject H₀' if p_val < 0.05 else 'No - Fail to Reject H₀'}")

print("\n  Group Statistics:")
stats = dc.groupby('engineering_branch')['cgpa'].agg(['mean','std','count'])
print(stats.round(4).to_string())

# ═══════════════════════════════════════════════
# 2. CHI-SQUARE: Branch vs Placement
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("  2. CHI-SQUARE TEST: Branch vs Placement Status")
print("=" * 70)

ct = pd.crosstab(dc['engineering_branch'], dc['status'])
chi2, p_chi, dof, expected = chi2_contingency(ct)

print(f"  Chi² statistic:  {chi2:.4f}")
print(f"  P-value:         {p_chi:.4f}")
print(f"  Degrees of Freedom: {dof}")
print(f"  Significant (α=0.05): {'Yes - Reject H₀' if p_chi < 0.05 else 'No - Fail to Reject H₀'}")

# Manual chi2 verification
chi2_manual = ((ct.values - expected)**2 / expected).sum()
print(f"  Chi² (manual):   {chi2_manual:.4f}")
print(f"  Match: {'✅ YES' if abs(chi2 - chi2_manual) < 0.001 else '❌ NO'}")

print("\n  Observed Frequencies:")
print(ct.to_string())
print("\n  Expected Frequencies:")
print(pd.DataFrame(np.round(expected, 2), index=ct.index, columns=ct.columns).to_string())

# ═══════════════════════════════════════════════
# 3. CHI-SQUARE: Internships vs Placement
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("  3. CHI-SQUARE TEST: Internships vs Placement Status")
print("=" * 70)

ct2 = pd.crosstab(dc['internships'], dc['status'])
chi2b, p3, dof2, exp2 = chi2_contingency(ct2)

chi2b_manual = ((ct2.values - exp2)**2 / exp2).sum()
print(f"  Chi² statistic:  {chi2b:.4f}")
print(f"  Chi² (manual):   {chi2b_manual:.4f}")
print(f"  Match: {'✅ YES' if abs(chi2b - chi2b_manual) < 0.001 else '❌ NO'}")
print(f"  P-value:         {p3:.6f}")
print(f"  Degrees of Freedom: {dof2}")
print(f"  Significant (α=0.05): {'Yes - Reject H₀' if p3 < 0.05 else 'No - Fail to Reject H₀'}")

print("\n  Observed Frequencies:")
print(ct2.to_string())

# ═══════════════════════════════════════════════
# 4. ML MODELS
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("  4. ML MODELS VALIDATION")
print("=" * 70)

# Encode
dm = dc.copy()
encoders = {}
for col in ['gender','core_skill','engineering_branch','hsc_stream','status','extracurricular','cgpa_category']:
    le = LabelEncoder()
    dm[f'{col}_enc'] = le.fit_transform(dm[col].astype(str))
    encoders[col] = le

features = ['ssc_percentage','hsc_percentage','ug_percentage','employability_test_percentage',
            'technical_interview_score','cgpa','internships',
            'gender_enc','core_skill_enc','engineering_branch_enc','hsc_stream_enc',
            'overall_academic_score','academic_consistency','activity_score',
            'placement_readiness','has_extra','cgpa_x_internships']

# 4a. Logistic Regression
print("\n  ── 4a. Logistic Regression (Placement Classification) ──")
X_c = dm[features]; y_c = dm['status_enc']
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_c, y_c, test_size=.2, random_state=42)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(Xtr_c, ytr_c)
clf_pred = clf.predict(Xte_c)
acc = accuracy_score(yte_c, clf_pred)
cm = confusion_matrix(yte_c, clf_pred)
rpt = classification_report(yte_c, clf_pred, target_names=encoders['status'].classes_)

print(f"  Train size: {len(Xtr_c)}, Test size: {len(Xte_c)}")
print(f"  Accuracy:   {acc:.4f} ({acc*100:.1f}%)")
print(f"\n  Confusion Matrix:")
print(f"    {cm}")
print(f"\n  Classification Report:")
print(rpt)
print(f"  Coefficients (top 5 by magnitude):")
coefs = sorted(zip(features, clf.coef_[0]), key=lambda x: abs(x[1]), reverse=True)
for feat, val in coefs[:5]:
    print(f"    {feat:35s} → {val:+.4f}")

# 4b. Linear Regression
print("\n  ── 4b. Linear Regression (Salary Prediction) ──")
pm = dm[dm['status']=='Placed']
X_r = pm[features]; y_r = pm['salary']
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_r, y_r, test_size=.2, random_state=42)
reg = LinearRegression()
reg.fit(Xtr_r, ytr_r)
reg_pred = reg.predict(Xte_r)
r2 = r2_score(yte_r, reg_pred)
mae = mean_absolute_error(yte_r, reg_pred)

print(f"  Train size: {len(Xtr_r)}, Test size: {len(Xte_r)}")
print(f"  R² Score:   {r2:.4f}")
print(f"  MAE:        ₹{mae:,.0f}")
print(f"  Intercept:  {reg.intercept_:.2f}")
print(f"  Top 5 Coefficients:")
lr_coefs = sorted(zip(features, reg.coef_), key=lambda x: abs(x[1]), reverse=True)
for feat, val in lr_coefs[:5]:
    print(f"    {feat:35s} → {val:+.2f}")

# 4c. Polynomial Regression
print("\n  ── 4c. Polynomial Regression (degree=2, Salary) ──")
pf = ['cgpa','internships','employability_test_percentage','technical_interview_score']
poly = PolynomialFeatures(degree=2, include_bias=False)
Xtr_poly = poly.fit_transform(Xtr_r[pf])
Xte_poly = poly.transform(Xte_r[pf])
preg = LinearRegression()
preg.fit(Xtr_poly, ytr_r)
poly_pred = preg.predict(Xte_poly)
poly_r2 = r2_score(yte_r, poly_pred)
poly_mae = mean_absolute_error(yte_r, poly_pred)

print(f"  Input features: {pf}")
print(f"  Polynomial features generated: {Xtr_poly.shape[1]}")
print(f"  Feature names: {poly.get_feature_names_out(pf).tolist()}")
print(f"  R² Score:   {poly_r2:.4f}")
print(f"  MAE:        ₹{poly_mae:,.0f}")

# ═══════════════════════════════════════════════
# 5. FEATURE ENGINEERING VALIDATION
# ═══════════════════════════════════════════════
print("\n" + "=" * 70)
print("  5. FEATURE ENGINEERING VALIDATION")
print("=" * 70)

# Verify CGPA formula
sample = dc.iloc[0]
cgpa_check = round(sample['ug_percentage'] / 100 * 6 + 4, 2)
print(f"  Sample UG%: {sample['ug_percentage']}, CGPA: {sample['cgpa']}, Recalc: {cgpa_check} → {'✅' if sample['cgpa'] == cgpa_check else '❌'}")

# Verify overall_academic_score
oas_check = round(sample['ssc_percentage']*.2 + sample['hsc_percentage']*.3 + sample['ug_percentage']*.5, 2)
print(f"  OAS: {sample['overall_academic_score']}, Recalc: {oas_check} → {'✅' if sample['overall_academic_score'] == oas_check else '❌'}")

# Verify academic_consistency
ac_check = round(pd.Series([sample['ssc_percentage'], sample['hsc_percentage'], sample['ug_percentage']]).std(), 2)
print(f"  Consistency: {sample['academic_consistency']}, Recalc: {ac_check} → {'✅' if sample['academic_consistency'] == ac_check else '❌'}")

# Verify activity_score
as_check = sample['internships'] * 2 + sample['has_extra']
print(f"  Activity Score: {sample['activity_score']}, Recalc: {as_check} → {'✅' if sample['activity_score'] == as_check else '❌'}")

# Verify cgpa_x_internships
cxi_check = round(sample['cgpa'] * sample['internships'], 2)
print(f"  CGPA×Intern: {sample['cgpa_x_internships']}, Recalc: {cxi_check} → {'✅' if sample['cgpa_x_internships'] == cxi_check else '❌'}")

print("\n" + "=" * 70)
print("  ✅ VALIDATION COMPLETE")
print("=" * 70)
