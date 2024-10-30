# AICore Finance EDA Project

_Started on 26/09/2024_

This project was set by AICore as part of the Data Analytics pathway. It focuses on performing Exploratory Data Analysis (EDA) on a loan payment data set stored in an AWS RDS database. The objective is to extract meaningful insights and prepare the dataset for further analysis by handling missing values, skewed data, outliers, and correlations, ultimately providing a clear understanding of loan performance and potential losses.

## **Scenario**

You currently work for a large financial institution, where managing loans is a critical component of business operations.

To ensure informed decisions are made about loan approvals and risk is efficiently managed, your task is to gain a comprehensive understanding of the loan portfolio data.

Your task is to perform exploratory data analysis on the loan portfolio, using various statistical and data visualisation techniques to uncover patterns, relationships, and anomalies in the loan data.

This information will enable the business to make more informed decisions about loan approvals, pricing, and risk management.

By conducting exploratory data analysis on the loan data, you aim to gain a deeper understanding of the risk and return associated with the business' loans.

Ultimately, your goal is to improve the performance and profitability of the loan portfolio.

---

## **File Structure & Naming Conventions**

- **Python Scripts**:
  - `db_utils.py`: Contains classes and methods for database connection and data extraction.
  - `db_converter.py`: Responsible for handling data format transformations.
  - `db_deleter.py`: Used to delete specific data entries and columns that aren't needed.
  - `db_info.py`: Utility functions to gather and summarize data information.
  - `db_plotter.py`: Visualizations for exploring data trends and analysis outcomes.
  - `db_transformer.py`: Used for data imputation and skewness rectification.

- **Configuration Files**:
  - `credentials.yaml`: Stores database credentials for secure access to AWS RDS.
  - `requirements.txt`: Contains a list of the conda packages needed for this project.

- **Output Data**:
  - `loan_payments_data.csv`: The original dataset from the RDS database
  - `pretransformed_loan_payments.csv`: The dataset after data imputation and data type correction.
  - `transformed_loan_payments.csv`: Dataset which includes skewness rectification

---

## **Handling the dataset**

The dataset was originally stored in the RDS database. The `db_utils.py` file makes use of sqlalchemy to store the database and then convert it into a csv file which can be used for this project.<br>

The next step involved creating a script which would deal with all the columns which were of the wrong data type - of which there were several. These involved converting many to categorical, some to datetime, some to objects and some to numerical. These were determined based on their future use throughout the project as well as the contents of the column. <br>

Next a script was created to provide an overview of the dataset. This included many statistical values as well as quantities such as null/zero percentages throughout the column as well as skewness. Skewness was identified within this column using both the boxcox and the yeo-johnson methods and depending on which one provided the best skewness rectification, that method will be the one being applied in the later script. A heatmap of correlations with the other columns were also identified.

A deleter script was also created which would allow us to remove columns and even specific entries. These were removed based on how correlated they are to other columns, how many zeroes and nulls they have and just in general, if they were a useful column to keep around. 

The transformer script is where most of this milestone happens. It is where data is imputed based on their data types, the values they contain and the contents of the column. This is also where the aforemention skewness correction happens.

Lastly, a plotter script was created to plot several graphs.

These were all scripts that were created and then imported into the noteboook: `Loan_Data_Analysis.ipynb` where the dataset was actually handled.
---

## **Performing EDA On The Dataset**

This EDA was carried out based on specific questions that were presented to me - in the practical sense these would be what the stakeholders would want to know and the following findings are on display within the `Loan_Data_Queries.ipynb` file.

### **Task 1: Current State of Loans**

1. **Loan Recovery Analysis**:
   - Calculate the percentage of recovered loans against the total loan amount, including interest.
   - Visualize the recovery rate and future recovery projections over a six-month term.

### **Task 2: Calculating Losses from Charged-Off Loans**

1. **Loss Evaluation**:
   - Identify loans marked as “Charged Off” in `loan_status` to calculate their percentage and the amount paid prior to default.

### **Task 3: Projected Loss on Charged-Off Loans**

1. **Expected Loss Calculation**:
   - Estimate lost revenue for “Charged Off” loans if they had completed their terms and visualize the projected loss over time.

### **Task 4: Evaluating Potential Loss from Late Payments**

1. **Late Payment Risks**:
   - Determine the percentage of late-paying customers and the financial risk if they default.
   - Calculate the impact on total expected revenue if all late-paying customers were to charge off.

### **Task 5: Indicators of Loan Default Risk**

1. **Identifying Default Indicators**:
   - Analyze attributes like loan grade, purpose, and home ownership to assess their correlation with default risk.
   - Compare these factors between loans marked as “Charged Off” and those at risk to identify predictive indicators.
   
--- 

## **License**
This project is licensed under AICore