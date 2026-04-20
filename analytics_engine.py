"""
Analytics Engine — Feature Engineering, ML Models, Statistical Tests
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, r2_score, mean_absolute_error,
                             mean_squared_error, confusion_matrix, classification_report)
from scipy.stats import chi2_contingency, f_oneway
import warnings
warnings.filterwarnings('ignore')


def load_data():
    df = pd.read_csv('data/Placement_Data.csv')
    df.rename(columns={
        'sl_no': 'serial_no', 'ssc_p': 'ssc_percentage', 'ssc_b': 'ssc_board',
        'hsc_p': 'hsc_percentage', 'hsc_b': 'hsc_board', 'hsc_s': 'hsc_stream',
        'etest_p': 'employability_test_pct',
    }, inplace=True)

    dc = df.copy()
    np.random.seed(42)

    # Engineering branches
    branches = ['Computer Engineering', 'Information Technology',
                'Electronics & Telecomm.', 'Mechanical Engineering', 'Civil Engineering']
    dc['engineering_branch'] = np.random.choice(branches, size=len(dc), p=[.35,.25,.20,.10,.10])

    dc['ug_percentage'] = dc['degree_p']
    dc.drop(columns=['degree_p', 'degree_t'], inplace=True)

    # Core skill
    dc['core_skill'] = dc['specialisation'].map({'Mkt&HR': 'Data Science & AI', 'Mkt&Fin': 'Full-Stack Dev'})
    dc.drop(columns=['specialisation'], inplace=True)

    dc['technical_interview_score'] = dc['mba_p']
    dc.drop(columns=['mba_p'], inplace=True)

    if 'workex' in dc.columns:
        dc.drop(columns=['workex'], inplace=True)

    # CGPA from ug_percentage (scale 4-10)
    dc['cgpa'] = np.round(dc['ug_percentage'] / 100 * 6 + 4, 2)

    # Internships (correlated with placement)
    placed_mask = dc['status'] == 'Placed'
    dc.loc[placed_mask, 'internships'] = np.random.choice([0,1,2,3], size=placed_mask.sum(), p=[.10,.25,.40,.25])
    dc.loc[~placed_mask, 'internships'] = np.random.choice([0,1,2,3], size=(~placed_mask).sum(), p=[.40,.35,.20,.05])
    dc['internships'] = dc['internships'].astype(int)

    # Extracurricular
    acts = ['Sports','Coding Club','Cultural','Robotics','None','Hackathons','NSS/NCC']
    dc.loc[placed_mask, 'extracurricular'] = np.random.choice(acts, size=placed_mask.sum(),
        p=[.12,.25,.10,.15,.08,.20,.10])
    dc.loc[~placed_mask, 'extracurricular'] = np.random.choice(acts, size=(~placed_mask).sum(),
        p=[.15,.10,.15,.05,.30,.10,.15])

    dc['salary'].fillna(0, inplace=True)
    dc.drop_duplicates(inplace=True)
    return dc


def feature_engineering(df):
    fe = df.copy()
    # 1. CGPA Category
    fe['cgpa_category'] = pd.cut(fe['cgpa'], bins=[0,5.5,6.5,7.5,8.5,10],
                                  labels=['Below Avg','Average','Good','Very Good','Excellent'])
    # 2. Academic consistency (lower std = more consistent)
    fe['academic_consistency'] = fe[['ssc_percentage','hsc_percentage','ug_percentage']].std(axis=1).round(2)
    # 3. Overall academic score (weighted)
    fe['overall_academic_score'] = (fe['ssc_percentage']*.2 + fe['hsc_percentage']*.3 + fe['ug_percentage']*.5).round(2)
    # 4. Activity score
    fe['has_extracurricular'] = (fe['extracurricular'] != 'None').astype(int)
    fe['activity_score'] = fe['internships'] * 2 + fe['has_extracurricular']
    # 5. Placement readiness index
    fe['placement_readiness'] = (
        fe['overall_academic_score'] * 0.4 +
        fe['employability_test_pct'] * 0.2 +
        fe['technical_interview_score'] * 0.2 +
        fe['activity_score'] * 3
    ).round(2)
    # 6. Interaction features
    fe['cgpa_x_internships'] = (fe['cgpa'] * fe['internships']).round(2)
    fe['cgpa_x_etest'] = (fe['cgpa'] * fe['employability_test_pct'] / 10).round(2)
    fe['cgpa_squared'] = (fe['cgpa'] ** 2).round(2)
    return fe


def train_all_models(df_fe):
    dm = df_fe.copy()
    encoders = {}
    for col in ['gender','core_skill','engineering_branch','hsc_stream','status','extracurricular','cgpa_category']:
        le = LabelEncoder()
        dm[f'{col}_enc'] = le.fit_transform(dm[col].astype(str))
        encoders[col] = le

    feats = ['ssc_percentage','hsc_percentage','ug_percentage','employability_test_pct',
             'technical_interview_score','cgpa','internships',
             'gender_enc','core_skill_enc','engineering_branch_enc','hsc_stream_enc',
             'overall_academic_score','academic_consistency','activity_score',
             'placement_readiness','has_extracurricular','cgpa_x_internships']

    # ── Logistic Regression ──
    X_c = dm[feats]; y_c = dm['status_enc']
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_c, y_c, test_size=.2, random_state=42)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(Xtr_c, ytr_c)
    clf_pred = clf.predict(Xte_c)
    clf_acc = accuracy_score(yte_c, clf_pred)
    clf_cm = confusion_matrix(yte_c, clf_pred)
    clf_rpt = classification_report(yte_c, clf_pred, target_names=encoders['status'].classes_, output_dict=True)

    # ── Linear Regression (salary) ──
    pm = dm[dm['status']=='Placed']
    X_r = pm[feats]; y_r = pm['salary']
    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_r, y_r, test_size=.2, random_state=42)
    reg = LinearRegression()
    reg.fit(Xtr_r, ytr_r)
    reg_pred = reg.predict(Xte_r)
    reg_r2 = r2_score(yte_r, reg_pred)
    reg_mae = mean_absolute_error(yte_r, reg_pred)

    # ── Polynomial Regression (salary, degree=2 on subset) ──
    poly_feats = ['cgpa','internships','employability_test_pct','technical_interview_score']
    Xtr_p = Xtr_r[poly_feats]; Xte_p = Xte_r[poly_feats]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    Xtr_poly = poly.fit_transform(Xtr_p)
    Xte_poly = poly.transform(Xte_p)
    preg = LinearRegression()
    preg.fit(Xtr_poly, ytr_r)
    poly_pred = preg.predict(Xte_poly)
    poly_r2 = r2_score(yte_r, poly_pred)
    poly_mae = mean_absolute_error(yte_r, poly_pred)

    return {
        'clf': clf, 'clf_acc': clf_acc, 'clf_cm': clf_cm, 'clf_rpt': clf_rpt,
        'reg': reg, 'reg_r2': reg_r2, 'reg_mae': reg_mae,
        'reg_pred': reg_pred, 'reg_actual': yte_r.values,
        'preg': preg, 'poly': poly, 'poly_r2': poly_r2, 'poly_mae': poly_mae,
        'poly_pred': poly_pred, 'poly_feats': poly_feats,
        'encoders': encoders, 'features': feats,
        'coef_lr': dict(zip(feats, reg.coef_.round(2))),
        'coef_logistic': dict(zip(feats, clf.coef_[0].round(4))),
    }


def run_stat_tests(df_fe):
    res = {}
    # ANOVA: CGPA across branches
    groups = [g['cgpa'].values for _, g in df_fe.groupby('engineering_branch')]
    f1, p1 = f_oneway(*groups)
    stats1 = df_fe.groupby('engineering_branch')['cgpa'].agg(['mean','std','count']).round(3)
    grand_mean = df_fe['cgpa'].mean()
    ssb = sum(len(g)*(g.mean()-grand_mean)**2 for _, g in df_fe.groupby('engineering_branch')['cgpa'])
    ssw = sum(((g - g.mean())**2).sum() for _, g in df_fe.groupby('engineering_branch')['cgpa'])
    k = df_fe['engineering_branch'].nunique()
    n = len(df_fe)
    res['anova_cgpa'] = {
        'title': 'ANOVA: CGPA across Engineering Branches',
        'H0': 'Mean CGPA is equal across all engineering branches',
        'H1': 'At least one branch has a different mean CGPA',
        'f': round(f1,4), 'p': round(p1,4), 'sig': p1<0.05,
        'stats': stats1, 'grand_mean': round(grand_mean,3),
        'SSB': round(ssb,3), 'SSW': round(ssw,3),
        'MSB': round(ssb/(k-1),3), 'MSW': round(ssw/(n-k),3),
        'df_between': k-1, 'df_within': n-k,
    }

    # Chi-Square: Branch vs Placement
    ct1 = pd.crosstab(df_fe['engineering_branch'], df_fe['status'])
    chi1, p2, dof1, exp1 = chi2_contingency(ct1)
    res['chi2_branch'] = {
        'title': 'Chi-Square: Branch vs Placement Status',
        'H0': 'Engineering branch and placement status are independent',
        'H1': 'There is an association between branch and placement',
        'chi2': round(chi1,4), 'p': round(p2,4), 'dof': dof1, 'sig': p2<0.05,
        'observed': ct1,
        'expected': pd.DataFrame(np.round(exp1,2), index=ct1.index, columns=ct1.columns),
    }

    # Chi-Square: Internships vs Placement
    ct2 = pd.crosstab(df_fe['internships'], df_fe['status'])
    chi2_val, p3, dof2, exp2 = chi2_contingency(ct2)
    res['chi2_intern'] = {
        'title': 'Chi-Square: Internships vs Placement Status',
        'H0': 'Number of internships and placement status are independent',
        'H1': 'There is an association between internships and placement',
        'chi2': round(chi2_val,4), 'p': round(p3,4), 'dof': dof2, 'sig': p3<0.05,
        'observed': ct2,
        'expected': pd.DataFrame(np.round(exp2,2), index=ct2.index, columns=ct2.columns),
    }

    # ANOVA: Salary across branches (placed only)
    placed = df_fe[df_fe['status']=='Placed']
    sg = [g['salary'].values for _, g in placed.groupby('engineering_branch')]
    f2, p4 = f_oneway(*sg)
    res['anova_salary'] = {
        'title': 'ANOVA: Salary across Engineering Branches',
        'H0': 'Mean salary is equal across all engineering branches',
        'H1': 'At least one branch has a different mean salary',
        'f': round(f2,4), 'p': round(p4,4), 'sig': p4<0.05,
        'stats': placed.groupby('engineering_branch')['salary'].agg(['mean','std','count']).round(0),
    }
    return res
