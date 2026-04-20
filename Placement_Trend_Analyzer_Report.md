# Placement Trend Analyzer — Project Report

## 1. Title Page
**Project Title:** Placement Trend Analyzer
**Subject:** Data Analytics
**Date:** April 2026

---

## 2. Abstract
The Placement Trend Analyzer is a data analytics mini-project designed to uncover key insights from a campus recruitment dataset. It provides an end-to-end data pipeline starting from data cleaning and exploratory data analysis (EDA) to building machine learning models for placement prediction and salary estimation. The project is bundled with an interactive Streamlit dashboard featuring glassmorphism design for intuitive visualization and filtering.

---

## 3. Introduction
Campus placements are a critical metric for both students and educational institutions. Understanding the trends and factors that influence placement probability and salary can help students focus on the right skills and help institutions improve their training programs. The objective of this project is to analyze these trends, visualize the impact of academic performance and specializations, and predict placement outcomes using machine learning.

---

## 4. Dataset Description
*   **Source:** Kaggle ("Factors Affecting Campus Placement" by Ben Roshan).
*   **Size:** 215 records (students) across 15 features.
*   **Features:**
    *   `gender`: Male / Female
    *   `ssc_percentage`, `ssc_board`: 10th grade percentage and board.
    *   `hsc_percentage`, `hsc_board`, `hsc_stream`: 12th grade details.
    *   `degree_percentage`, `degree_type`: Undergraduate details.
    *   `work_experience`: Yes / No
    *   `employability_test`: Employability test percentage.
    *   `specialisation`: MBA specialization (Mkt&Fin, Mkt&HR).
    *   `mba_percentage`: MBA percentage.
    *   `status`: Placed / Not Placed
    *   `salary`: Salary offered (if placed).

---

## 5. Data Cleaning
During the data cleaning phase, the following steps were performed:
1.  **Handling Missing Values:** The `salary` column contained missing values for students who were "Not Placed". These were filled with `0` since unplaced students do not have a salary.
2.  **Duplicate Check:** The dataset was checked for exact duplicate rows (none were found, ensuring data integrity).
3.  **Renaming Columns:** Column names were updated (e.g., `ssc_p` to `ssc_percentage`, `sl_no` to `serial_no`) to make the data more accessible and readable.

---

## 6. Exploratory Data Analysis (EDA)
Comprehensive EDA was performed and 8 distinct visualizations were generated:
1.  **Placement Rate:** 148 out of 215 students (68.8%) got placed.
2.  **Salary Distribution:** Most salaries for placed students fall between ₹2L and ₹3.5L, with an average salary of ~₹2.88L.
3.  **Branch-wise Placement:** Commerce and Management (`Comm&Mgmt`) streams boast the highest overall student numbers and placements.
4.  **Gender-wise Placement:** Male students form the majority of the cohort, but both genders show proportionally similar placement success.
5.  **Employability Test vs. Salary:** Only a weak positive correlation exists between high employability test scores and higher salary brackets.
6.  **Correlation Heatmap:** Moderate positive correlation (0.51) identified between 10th and 12th-grade percentages. Salary poorly correlates directly with academic percentages overall.
7.  **Work Experience Impact:** Students with previous work experience command a slightly higher median salary.
8.  **MBA Specialisation:** Marketing & Finance (`Mkt&Fin`) students show slightly higher placement rates compared to Marketing & HR (`Mkt&HR`).

---

## 7. Machine Learning Prediction
Two machine learning models were trained to predict student outcomes:
1.  **Placement Prediction (Logistic Regression):**
    *   **Features used:** 10th %, 12th %, Degree %, E-Test %, MBA %, Gender, Work Experience, Specialization, Degree Type, HSC Stream.
    *   **Target:** `status` (Placed / Not Placed)
    *   **Accuracy Achieved:** ~88.4%
2.  **Salary Prediction (Linear Regression):**
    *   **Target:** `salary` (trained only on the placed student subset).
    *   **Result:** Exhibited larger variance (reflected in MAE), as salary is highly dependent on qualitative factors beyond standardized scores.

---

## 8. Dashboard
An interactive dashboard was constructed using **Streamlit** and **Plotly**. Real-time filters and ML inference engines were deployed on the web interface, utilizing a custom "Glassmorphism" light theme to achieve a modern and polished aesthetic. Users can predict outcomes by adjusting sliders for academic scores.

---

## 9. Conclusion
The project verifies that while strong academic performance is a baseline, a specialization in Marketing & Finance coupled with prior Work Experience provides compounding advantages in both placement success and initial salary. The combined analytical and predictive dashboard serves as a comprehensive tool for campus recruitment analysis.

---

## 10. References
1.  Kaggle Dataset: "Factors Affecting Campus Placement" (Ben Roshan)
2.  Python Documentation, Pandas, Scikit-learn, Streamlit, and Plotly Official Guides.
