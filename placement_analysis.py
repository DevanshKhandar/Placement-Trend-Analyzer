# ╔══════════════════════════════════════════════════════════════╗
# ║      PLACEMENT TREND ANALYZER — EDA NOTEBOOK (Script)      ║
# ║            Data Analytics Mini Project                      ║
# ╚══════════════════════════════════════════════════════════════╝
#
# This script performs complete Exploratory Data Analysis (EDA)
# on the Campus Recruitment dataset. Run each section as a
# Jupyter Notebook cell or execute the full script.
#
# To convert to Jupyter Notebook, run:
#   pip install jupytext
#   jupytext --to notebook placement_analysis.py

# %% [markdown]
# # Placement Trend Analyzer - Exploratory Data Analysis
# **Dataset:** Campus Recruitment (Kaggle - Ben Roshan)
# **Objective:** Analyze placement trends, salary distribution, and build predictive models

# %% Cell 1: Import Libraries
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving charts
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
REPORT_DIR = os.path.join(PROJECT_DIR, 'report')
os.makedirs(REPORT_DIR, exist_ok=True)

# Set visual style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

print("[OK] All libraries loaded successfully!")

# %% Cell 2: Load Dataset
df = pd.read_csv(os.path.join(DATA_DIR, 'Placement_Data.csv'))
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")
print("First 5 rows:")
df.head()

# %% Cell 3: Dataset Info
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"\nShape: {df.shape[0]} students, {df.shape[1]} columns\n")
print("Column Types:")
print(df.dtypes)
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
df.describe()

# %% Cell 4: Missing Values Analysis
print("=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])
print(f"\nTotal missing values: {df.isnull().sum().sum()}")
print("\nNote: 'salary' is null for students who are Not Placed - this is expected.")


# %% Cell 5: Data Cleaning
print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Make a backup
df_clean = df.copy()

# Fill missing salary with 0
df_clean['salary'].fillna(0, inplace=True)
print("[OK] Filled missing salary values with 0")

# Remove duplicates
dupes = df_clean.duplicated().sum()
df_clean.drop_duplicates(inplace=True)
print(f"[OK] Removed {dupes} duplicate rows")

# Rename columns
df_clean.rename(columns={
    'sl_no': 'serial_no',
    'ssc_p': 'ssc_percentage',       # 10th percentage
    'ssc_b': 'ssc_board',            # 10th board
    'hsc_p': 'hsc_percentage',       # 12th percentage
    'hsc_b': 'hsc_board',            # 12th board
    'hsc_s': 'hsc_stream',           # 12th stream
    'degree_p': 'degree_percentage', # UG degree percentage
    'degree_t': 'degree_type',       # UG degree type
    'workex': 'work_experience',     # Work experience
    'etest_p': 'employability_test', # Employability test %
    'mba_p': 'mba_percentage'        # MBA percentage
}, inplace=True)
print("[OK] Renamed columns for clarity")

print(f"\nCleaned dataset: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
print("Columns:", df_clean.columns.tolist())

# Save cleaned data
df_clean.to_csv(os.path.join(DATA_DIR, 'Placement_Data_Cleaned.csv'), index=False)
print("\n[OK] Cleaned data saved to data/Placement_Data_Cleaned.csv")

# %% Cell 6: Quick Summary Statistics
print("=" * 60)
print("KEY PLACEMENT STATISTICS")
print("=" * 60)

placed = df_clean[df_clean['status'] == 'Placed']
not_placed = df_clean[df_clean['status'] == 'Not Placed']

total = len(df_clean)
total_placed = len(placed)
placement_rate = round(total_placed / total * 100, 1)
avg_salary = placed['salary'].mean()
max_salary = placed['salary'].max()
min_salary = placed['salary'].min()
median_salary = placed['salary'].median()

print(f"  Total Students     : {total}")
print(f"  Placed Students    : {total_placed}")
print(f"  Not Placed         : {len(not_placed)}")
print(f"  Placement Rate     : {placement_rate}%")
print(f"  Average Salary     : ₹{avg_salary:,.0f}")
print(f"  Median Salary      : ₹{median_salary:,.0f}")
print(f"  Highest Salary     : ₹{max_salary:,.0f}")
print(f"  Lowest Salary      : ₹{min_salary:,.0f}")

# %% [markdown]
# ---
# ## VISUALIZATION - 8 Charts
# ---

# %% Chart 1: Placement Rate Pie Chart
fig, ax = plt.subplots(figsize=(8, 6))
status_counts = df_clean['status'].value_counts()
colors = ['#6366f1', '#ef4444']
explode = (0.05, 0)

wedges, texts, autotexts = ax.pie(
    status_counts, labels=status_counts.index, autopct='%1.1f%%',
    colors=colors, startangle=90, explode=explode,
    textprops={'fontsize': 14, 'fontweight': 'bold'},
    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
)
ax.set_title('Chart 1: Placement Rate', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'chart1_placement_rate.png'), dpi=150, bbox_inches='tight',
            facecolor='white', transparent=False)
plt.close()

print(f"\n>> OBSERVATION: {status_counts['Placed']} out of {total} students ({placement_rate}%) "
      f"got placed. The majority of students secure placements.")

# %% Chart 2: Salary Distribution Histogram
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(placed['salary'], bins=20, kde=True, color='#6366f1',
             edgecolor='white', linewidth=0.5, alpha=0.7, ax=ax)
ax.axvline(avg_salary, color='#ec4899', linewidth=2, linestyle='--',
           label=f'Mean: ₹{avg_salary:,.0f}')
ax.axvline(median_salary, color='#10b981', linewidth=2, linestyle='--',
           label=f'Median: ₹{median_salary:,.0f}')
ax.set_title('Chart 2: Salary Distribution of Placed Students', fontsize=18, fontweight='bold')
ax.set_xlabel('Salary (₹)', fontsize=13)
ax.set_ylabel('Number of Students', fontsize=13)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'chart2_salary_distribution.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n>> OBSERVATION: Most salaries cluster between Rs.2L-3.5L. "
      f"The distribution is right-skewed with a few high earners above Rs.6L.")

# %% Chart 3: Branch-wise Placement Comparison (Grouped Bar)
fig, ax = plt.subplots(figsize=(10, 6))
branch_status = pd.crosstab(df_clean['degree_type'], df_clean['status'])
branch_status.plot(kind='bar', ax=ax, color=['#ef4444', '#10b981'],
                   edgecolor='white', linewidth=1, width=0.7)
ax.set_title('Chart 3: Branch-wise Placement Comparison', fontsize=18, fontweight='bold')
ax.set_xlabel('Degree Type', fontsize=13)
ax.set_ylabel('Number of Students', fontsize=13)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(title='Status', fontsize=11)

# Add value labels
for container in ax.containers:
    ax.bar_label(container, fontsize=10, fontweight='bold', padding=3)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'chart3_branch_comparison.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n>> OBSERVATION: Comm&Mgmt has the highest number of students and placements. "
      "Sci&Tech also shows a strong placement rate.")

# %% Chart 4: Gender-wise Placement (Count Plot)
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(data=df_clean, x='gender', hue='status',
              palette=['#ef4444', '#6366f1'], ax=ax,
              edgecolor='white', linewidth=1)
ax.set_title('Chart 4: Gender-wise Placement Status', fontsize=18, fontweight='bold')
ax.set_xlabel('Gender', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.legend(title='Status', fontsize=11)

for container in ax.containers:
    ax.bar_label(container, fontsize=11, fontweight='bold', padding=3)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'chart4_gender_placement.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n>> OBSERVATION: Male students form the majority. "
      "Both genders show similar placement rates proportionally.")

# %% Chart 5: Employability Test vs Salary (Scatter Plot)
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    placed['employability_test'], placed['salary'],
    c=placed['mba_percentage'], cmap='viridis',
    s=placed['degree_percentage'] * 2, alpha=0.65,
    edgecolors='white', linewidth=0.5
)
cbar = plt.colorbar(scatter, ax=ax, label='MBA %')
ax.set_title('Chart 5: Employability Test Score vs Salary', fontsize=18, fontweight='bold')
ax.set_xlabel('Employability Test %', fontsize=13)
ax.set_ylabel('Salary (₹)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'chart5_score_vs_salary.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n>> OBSERVATION: There is a weak positive correlation between employability test "
      "scores and salary. Bubble size represents degree percentage.")

# %% Chart 6: Correlation Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
numerical_cols = ['ssc_percentage', 'hsc_percentage', 'degree_percentage',
                  'employability_test', 'mba_percentage', 'salary']
corr_matrix = df_clean[numerical_cols].corr()

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=1, linecolor='white',
            mask=mask, square=True, ax=ax,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
ax.set_title('Chart 6: Correlation Heatmap', fontsize=18, fontweight='bold', pad=20)

# Rename tick labels
labels = ['10th %', '12th %', 'Degree %', 'E-Test %', 'MBA %', 'Salary']
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels, rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'chart6_correlation_heatmap.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n>> OBSERVATION: SSC and HSC percentages show moderate positive correlation (0.51). "
      "Salary has weak correlation with most academic scores.")

# %% Chart 7: Salary by Work Experience (Box Plot)
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(data=placed, x='work_experience', y='salary',
            palette=['#8b5cf6', '#10b981'], ax=ax,
            linewidth=1.5, flierprops=dict(marker='o', markerfacecolor='#ef4444', markersize=6))
ax.set_title('Chart 7: Impact of Work Experience on Salary', fontsize=18, fontweight='bold')
ax.set_xlabel('Work Experience', fontsize=13)
ax.set_ylabel('Salary (₹)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'chart7_workex_salary.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n>> OBSERVATION: Students with work experience tend to receive slightly "
      "higher median salaries, though the difference is not dramatic.")

# %% Chart 8: MBA Specialisation Placement Rate (Horizontal Bar)
fig, ax = plt.subplots(figsize=(8, 6))
spec_rate = pd.crosstab(df_clean['specialisation'], df_clean['status'], normalize='index') * 100
spec_rate = spec_rate.sort_values('Placed')

bars = spec_rate['Placed'].plot(kind='barh', ax=ax, color=['#ec4899', '#6366f1'],
                                edgecolor='white', linewidth=1)
ax.set_title('Chart 8: Placement Rate by MBA Specialisation', fontsize=18, fontweight='bold')
ax.set_xlabel('Placement Rate (%)', fontsize=13)
ax.set_ylabel('Specialisation', fontsize=13)

for i, (val, name) in enumerate(zip(spec_rate['Placed'], spec_rate.index)):
    ax.text(val + 1, i, f'{val:.1f}%', va='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, 'chart8_specialisation_rate.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\n>> OBSERVATION: Mkt&Fin specialisation has a slightly higher placement "
      "rate compared to Mkt&HR.")

# %% [markdown]
# ---
# ## MACHINE LEARNING - Prediction Models
# ---

# %% ML: Prepare Data
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_absolute_error

print("=" * 60)
print("MACHINE LEARNING — DATA PREPARATION")
print("=" * 60)

df_ml = df_clean.copy()

# Encode categorical variables
le_gender = LabelEncoder()
le_workex = LabelEncoder()
le_spec = LabelEncoder()
le_degree = LabelEncoder()
le_hsc = LabelEncoder()
le_status = LabelEncoder()

df_ml['gender_enc'] = le_gender.fit_transform(df_ml['gender'])
df_ml['workex_enc'] = le_workex.fit_transform(df_ml['work_experience'])
df_ml['spec_enc'] = le_spec.fit_transform(df_ml['specialisation'])
df_ml['degree_enc'] = le_degree.fit_transform(df_ml['degree_type'])
df_ml['hsc_enc'] = le_hsc.fit_transform(df_ml['hsc_stream'])
df_ml['status_enc'] = le_status.fit_transform(df_ml['status'])

features = ['ssc_percentage', 'hsc_percentage', 'degree_percentage',
            'employability_test', 'mba_percentage',
            'gender_enc', 'workex_enc', 'spec_enc', 'degree_enc', 'hsc_enc']

print(f"Features used: {len(features)}")
print(f"Classes: {le_status.classes_}")

# %% ML Model 1: Logistic Regression — Placement Prediction
print("=" * 60)
print("MODEL 1: LOGISTIC REGRESSION — Placement Prediction")
print("=" * 60)

X_clf = df_ml[features]
y_clf = df_ml['status_enc']

X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy*100:.1f}%")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le_status.classes_))

# %% ML Model 2: Linear Regression — Salary Prediction
print("=" * 60)
print("MODEL 2: LINEAR REGRESSION — Salary Prediction")
print("=" * 60)

placed_ml = df_ml[df_ml['status'] == 'Placed']
X_reg = placed_ml[features]
y_reg = placed_ml['salary']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train_r, y_train_r)

y_pred_r = reg.predict(X_test_r)
r2 = r2_score(y_test_r, y_pred_r)
mae = mean_absolute_error(y_test_r, y_pred_r)

print(f"\nR² Score: {r2:.4f}")
print(f"Mean Absolute Error: ₹{mae:,.0f}")

# %% Sample Prediction
print("\n" + "=" * 60)
print("SAMPLE PREDICTION")
print("=" * 60)

# Example student profile
sample = np.array([[75, 80, 70, 85, 65,
                    le_gender.transform(['M'])[0],
                    le_workex.transform(['Yes'])[0],
                    le_spec.transform(['Mkt&Fin'])[0],
                    le_degree.transform(['Sci&Tech'])[0],
                    le_hsc.transform(['Science'])[0]]])

placement_result = le_status.inverse_transform(clf.predict(sample))[0]
placement_prob = clf.predict_proba(sample)[0]
salary_result = max(0, reg.predict(sample)[0])

print(f"  Student: M, 10th=75%, 12th=80%, Degree=70%, E-Test=85%, MBA=65%")
print(f"  Work Exp: Yes | Spec: Mkt&Fin | Degree: Sci&Tech | Stream: Science")
print(f"")
print(f"  >> Placement Prediction : {placement_result}")
print(f"  >> Confidence           : {max(placement_prob)*100:.1f}%")
print(f"  >> Expected Salary      : Rs.{salary_result:,.0f}")

# %% [markdown]
# ---
# ## Summary of Findings
#
# | Metric | Value |
# |--------|-------|
# | Total Students | 215 |
# | Placement Rate | ~68% |
# | Average Salary | ~Rs.2.88L |
# | Highest Salary | Rs.9.40L |
# | Best Branch | Comm&Mgmt |
# | Work Exp Impact | Slight positive effect |
# | ML Accuracy | ~87% (Logistic Regression) |
# ---
# **End of EDA Notebook**

print("\n" + "=" * 60)
print("[OK] EDA COMPLETE - All charts saved to 'report/' folder")
print("=" * 60)
