# AI-Core Finance EDA Project
Started on 26/09/2024

This project was set by AICore as part of the Data Analytics pathway. It focuses on performing Exploratory Data Analysis (EDA) on a loan payment data set stored in an AWS RDS database. The objective is to extract meaningful insights and prepare the dataset for further analysis by handling missing values, skewed data, outliers, and correlations, ultimately providing a clear understanding of loan performance and potential losses.

## Scenario
You currently work for a large financial institution, where managing loans is a critical component of business operations.

To ensure informed decisions are made about loan approvals and risk is efficiently managed, your task is to gain a comprehensive understanding of the loan portfolio data.

Your task is to perform exploratory data analysis on the loan portfolio, using various statistical and data visualisation techniques to uncover patterns, relationships, and anomalies in the loan data.

This information will enable the business to make more informed decisions about loan approvals, pricing, and risk management.

By conducting exploratory data analysis on the loan data, you aim to gain a deeper understanding of the risk and return associated with the business' loans.

Ultimately, your goal is to improve the performance and profitability of the loan portfolio.

## Table of Contents
[Dataset Schema](#dataset-schema)
[Scripts](#scripts)
[Import](#imports)
[Data Extraction](#data-extraction)

## Dataset Schema

Below you will find a data dictionary which describes each of the columns so you can understand the values and the purpose each one holds:
- **id**: unique id of the loan
- **member_id**: id of the member to took out the loan
- **loan_amount**: amount of loan the applicant received
- **funded_amount**: The total amount committed to the loan at the point in time 
- **funded_amount_inv**: The total amount committed by investors for that loan at that point in time 
- **term**: The number of monthly payments for the loan
- **int_rate**: Interest rate on the loan
- **instalment**: The monthly payment owned by the borrower
- **grade**: LC assigned loan grade
- **sub_grade**: LC assigned loan sub grade
- **employment_length**: Employment length in years.
- **home_ownership**: The home ownership status provided by the borrower
- **annual_inc**: The annual income of the borrower
- **verification_status**: Indicates whether the borrowers income was verified by the LC or the income source was verified
- **issue_date:** Issue date of the loan
- **loan_status**: Current status of the loan
- **payment_plan**: Indicates if a payment plan is in place for the loan. Indication borrower is struggling to pay.
- **purpose**: A category provided by the borrower for the loan request.
- **dti**: A ratio calculated using the borrowerâ€™s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrowerâ€™s self-reported monthly income.
- **delinq_2yr**: The number of 30+ days past-due payment in the borrower's credit file for the past 2 years.
- **earliest_credit_line**: The month the borrower's earliest reported credit line was opened
- **inq_last_6mths**: The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
- **mths_since_last_record**: The number of months since the last public record.
- **open_accounts**: The number of open credit lines in the borrower's credit file.
- **total_accounts**: The total number of credit lines currently in the borrower's credit file
- **out_prncp**: Remaining outstanding principal for total amount funded
- **out_prncp_inv**: Remaining outstanding principal for portion of total amount funded by investors
- **total_payment**: Payments received to date for total amount funded
- **total_rec_int**: Interest received to date
- **total_rec_late_fee**: Late fees received to date
- **recoveries**: post charge off gross recovery
- **collection_recovery_fee**: post charge off collection fee
- **last_payment_date**: Last month payment was received
- **last_payment_amount**: Last total payment amount received
- **next_payment_date**: Next scheduled payment date
- **last_credit_pull_date**: The most recent month LC pulled credit for this loan
- **collections_12_mths_ex_med**: Number of collections in 12 months excluding medical collections
- **mths_since_last_major_derog**: Months since most recent 90-day or worse rating
- **policy_code**: publicly available policy_code=1 new products not publicly available policy_code=2
- **application_type**: Indicates whether the loan is an individual application or a joint application with two co-borrowers

## Scripts

- **db_utils.py**: This is the main script, where the initial EDA happens. Here the data is being extracted from the AWS RDS database and loaded into a Pandas dataframe for further analysis. It contains functions which clean and preprocess the data, ensuring correct formatting for the further analysis. This script also identifies if there are any missing values or skewness present amongst the data which may throw off the subsequent plots and analysis being carried out, and after having identified the problematic columns, it will then handle any missing values and deal with the skewness.

- **db_query.py**: Inside this script is where further querying of the dataset happens. The purpose of this script is to answer any questions management has and dive deeper into the dataset to identify any patterns or trends not visible by the previous analysis carried out. After gaining these insights, the company will stand on much better ground to make better educated decisions on which loans to target and which pose huge risks on the company.

## Imports
Here are a list of the imports necessary for both scripts:

- **db_utils.py**:
```py
import pandas as pd
import numpy as np
import yaml
from sqlalchemy import create_engine, inspect
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import psycopg2
import os
```

- **db_query.py**:
```py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
```
## Data Extraction
As mentioned previously, the loan dataset used in this project is located in an AWS RDS database. Using the credentials provided and storing them in a YAML file, we can set up connector class.  which will load the dataset into a Pandas dataframe, using csv as an intermediary. 

```py
def load_creds(file_path):
    with open(file_path, 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials

class RDSDatabaseConnector:
    def __init__(self, RDS_USER, RDS_PASSWORD, RDS_HOST, RDS_PORT, RDS_DATABASE):
        self.credentials = {
            'rds_user': RDS_USER,
            'rds_password': RDS_PASSWORD,
            'rds_host': RDS_HOST,
            'rds_port': RDS_PORT,
            'rds_database': RDS_DATABASE
        }
        self.engine = None
    
    def initialize_engine(self):
        # Construct the database connection URL for PostgreSQL
        db_url = f"postgresql+psycopg2://{self.credentials['rds_user']}:" \
                f"{self.credentials['rds_password']}@{self.credentials['rds_host']}:" \
                f"{self.credentials['rds_port']}/{self.credentials['rds_database']}"
        self.engine = create_engine(db_url)

    def extract_data(self):
        if self.engine is None:
            raise Exception("Database engine not initialized. Call 'initialize_engine()' first.")
        
        query = "SELECT * FROM loan_payments;"
        with self.engine.connect() as connection:
            df = pd.read_sql(query, connection)
        return df

    def save_data(self, df, file_path):
        df.to_csv(file_path, index=False)

    def load_csv_to_dataframe(self, file_path):
        
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            print(f"The file at {file_path} was not found.")
            return None
        except pd.errors.EmptyDataError:
            print("The file is empty.")
            return None
        except pd.errors.ParserError:
            print("Error parsing the file.")
            return None
```