End-to-End Credit Risk & Portfolio Analysis Dashboard
► Project Overview
This project presents a comprehensive Business Intelligence solution designed to analyze and monitor the health of a large-scale loan portfolio. Leveraging a dataset of over 10.4 million loans valued at $16 billion, this dashboard provides a multi-faceted view of credit risk, identifies key drivers of default, and equips executive stakeholders with a dynamic tool for strategic decision-making and policy simulation.

The entire analytics lifecycle is covered, from initial data conversion and cleaning, through advanced feature engineering in Python, to the final development of an interactive, multi-page Power BI report.

► Dashboard Preview
Page 1: Credit Risk Executive Summary
The Executive Summary page provides a high-level, at-a-glance overview of key performance indicators (KPIs), portfolio trends over time, and a risk concentration matrix to instantly identify problematic loan segments.

Page 2: Risk Driver Analysis
The Risk Driver Analysis page is designed for deep-dive exploration. It features an AI-powered Decomposition Tree for interactive root cause analysis and a powerful 'What-If' simulation panel to model the financial impact of changes to lending policies.

► Data Preparation & Engineering Pipeline
The foundation of this dashboard is a robust, multi-step data preparation pipeline built with Python (Pandas).

Data Source & Initial Format
The dataset used for this project is the Lending Club Loan Data on Kaggle. The full dataset is provided in the .feather format, a fast, binary format that is not directly readable by Power BI. The first step was to convert this data into a usable .csv format.

Step 1: Data Conversion (.feather to .csv)
A Python script was used to read the large .feather file and convert it into a standard CSV format.

<details>
<summary>Click to view Data Conversion Python Script</summary>

import pandas as pd

# The name of the file you want to convert
feather_file_name = r'lending_club_clean.feather'

# The name you want for your new CSV file
csv_file_name = 'lending_club_full_dataset.csv'

# --- Script Starts Here ---
print(f"Starting conversion of: {feather_file_name}")
print("Please be patient, this may take a few minutes...")

try:
    # Read the data from the feather file
    df = pd.read_feather(feather_file_name)

    print("Feather file loaded successfully. Now writing to CSV...")

    # Write the data to a CSV file
    # index=False is important to avoid an extra column in your CSV
    df.to_csv(csv_file_name, index=False)

    print("-" * 30)
    print(f"SUCCESS! Your new file is ready: {csv_file_name}")
    print("-" * 30)

except FileNotFoundError:
    print(f"ERROR: The file '{feather_file_name}' was not found in this folder.")
    print("Please make sure the script is in the same folder as the data file.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

</details>

Step 2: Data Cleaning & Feature Engineering
Once in CSV format, the data was processed to clean inconsistencies, correct data types, prune irrelevant columns, and engineer new, high-value features critical for the analysis.

<details>
<summary>Click to view Data Cleaning & Feature Engineering Python Script</summary>

import pandas as pd
import numpy as np

def refine_and_engineer_features():
    """
    Transforms the raw Lending Club CSV into a refined, feature-rich dataset
    ready for advanced business intelligence and analysis.
    """
    # --- Configuration ---
    csv_file_path = r"lending_club_full_dataset.csv"
    output_path = r"lending_club_analysis_ready.csv"

    # --- Step 1: Load Data ---
    print(f"Loading raw data from {csv_file_path}...")
    df = pd.read_csv(csv_file_path, low_memory=False)
    print(f"Successfully loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    # --- Step 2: Early Column Pruning for Performance ---
    relevant_columns = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
        'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
        'issue_d', 'loan_status', 'purpose', 'addr_state', 'dti', 'delinq_2yrs',
        'earliest_cr_line', 'fico_range_low', 'open_acc', 'pub_rec',
        'revol_bal', 'revol_util', 'total_acc', 'last_pymnt_d',
        'pub_rec_bankruptcies', 'debt_settlement_flag'
    ]
    df = df[relevant_columns]
    print(f"Pruned dataset to {df.shape[1]} relevant columns.")

    # --- Step 3: Basic Cleaning and Type Conversion ---
    print("Performing basic data cleaning and type conversions...")
    df['term'] = df['term'].astype(str).str.extract(r'(\d+)').astype(int)
    df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')
    df['revol_util'] = pd.to_numeric(df['revol_util'], errors='coerce')
    df['issue_date'] = pd.to_datetime(df['issue_d'], format='%d-%m-%Y', errors='coerce')
    df['earliest_credit_line'] = pd.to_datetime(df['earliest_cr_line'], format='%d-%m-%Y', errors='coerce')
    df['last_payment_date'] = pd.to_datetime(df['last_pymnt_d'], format='%d-%m-%Y', errors='coerce')
    df['emp_length_years'] = df['emp_length'].astype(str) \
        .replace({'< 1 year': '0 years', '10+ years': '10 years', 'n/a': '0 years'}) \
        .str.extract(r'(\d+)').astype(float)
    df['loan_status_group'] = 'In Progress'
    df.loc[df['loan_status'] == 'Fully Paid', 'loan_status_group'] = 'Paid Off'
    df.loc[df['loan_status'] == 'Charged Off', 'loan_status_group'] = 'Default'
    numeric_cols = [
        'loan_amnt', 'annual_inc', 'dti', 'fico_range_low', 'open_acc',
        'pub_rec', 'revol_bal', 'total_acc', 'pub_rec_bankruptcies'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Step 4: Advanced Feature Engineering ---
    print("Engineering new features for deeper analysis...")
    df['loan_vintage_year'] = df['issue_date'].dt.year
    df['loan_vintage_month'] = df['issue_date'].dt.to_period('M')
    df['credit_history_months'] = (df['issue_date'].dt.year - df['earliest_credit_line'].dt.year) * 12 + \
                                  (df['issue_date'].dt.month - df['earliest_credit_line'].dt.month)
    fico_bins = [0, 639, 699, 749, 900]
    fico_labels = ['Poor', 'Fair', 'Good', 'Excellent']
    df['fico_bucket'] = pd.cut(df['fico_range_low'], bins=fico_bins, labels=fico_labels)

    # --- Step 5: Handle Missing Values ---
    print("Handling missing values...")
    df = df.fillna({
        'emp_length_years': 0, 'dti': 0.0, 'revol_util': 0.0,
        'pub_rec_bankruptcies': 0, 'credit_history_months': 0
    })
    df.dropna(subset=['issue_date', 'loan_amnt'], inplace=True)

    # --- Step 6: Final Column Selection and Save ---
    print("Selecting final columns and preparing to save...")
    final_columns = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'purpose',
        'emp_length_years', 'home_ownership', 'annual_inc', 'verification_status', 'addr_state', 'dti',
        'delinq_2yrs', 'fico_range_low', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
        'pub_rec_bankruptcies', 'issue_date', 'last_payment_date', 'loan_status_group', 'debt_settlement_flag',
        'loan_vintage_year', 'loan_vintage_month', 'credit_history_months', 'fico_bucket'
    ]
    df_final = df[final_columns].copy()
    df_final['loan_vintage_month'] = df_final['loan_vintage_month'].astype(str)
    df_final.to_csv(output_path, index=False)

    print("-" * 50)
    print(f"SUCCESS: Refined and engineered dataset saved to: {output_path}")
    print(f"Final dataset has {df_final.shape[0]} rows and {df_final.shape[1]} columns.")
    print("-" * 50)

if __name__ == "__main__":
    refine_and_engineer_features()

</details>

Step 3: Final Cleaned Dataset
For ease of use and to allow others to replicate the analysis, the final, analysis-ready CSV file (lending_club_analysis_ready.csv) is provided in this repository.

Click here to download the cleaned dataset

► Key Features & Analytics
Executive KPIs: Monitors Total Portfolio Value ($16.08bn), Loan Volume (10.4M), Overall Default Rate (11.51%), and Average Borrower FICO Score (704).

Vintage Analysis: Tracks portfolio growth against default rates by the year the loan was issued, enabling powerful cohort analysis.

Risk Hotspot Matrix: A conditionally formatted matrix that visually identifies high-risk intersections of loan purpose and credit grade.

Interactive Root Cause Analysis: An AI-powered Decomposition Tree allows users to dynamically drill down and find the primary drivers of loan defaults.

'What-If' Scenario Modeling: A dynamic simulation tool that allows executives to model the financial impact of changing the minimum FICO score for lending. This feature directly links analytics to business strategy and quantifies potential savings (e.g., $1.94bn at an 800 FICO minimum).

► Technical Stack
Data Cleaning & Feature Engineering: Python (Pandas)

Data Visualization & BI: Microsoft Power BI

Data Modeling & Calculations: DAX (Data Analysis Expressions)

► Contact
Author: Rushikesh Bedre

LinkedIn: https://www.linkedin.com/in/rushikesh-bedre/

GitHub: https://github.com/rushibedre

Email: rushibedre10@gmail.com
