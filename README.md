# AICore Finance EDA Project
_Started on 26/09/2024_

This project was set by AICore as part of the Data Analytics pathway. It focuses on performing Exploratory Data Analysis (EDA) on a loan payment data set stored in an AWS RDS database. The objective is to extract meaningful insights and prepare the dataset for further analysis by handling missing values, skewed data, outliers, and correlations, ultimately providing a clear understanding of loan performance and potential losses.

## Scenario
You currently work for a large financial institution, where managing loans is a critical component of business operations.

To ensure informed decisions are made about loan approvals and risk is efficiently managed, your task is to gain a comprehensive understanding of the loan portfolio data.

Your task is to perform exploratory data analysis on the loan portfolio, using various statistical and data visualisation techniques to uncover patterns, relationships, and anomalies in the loan data.

This information will enable the business to make more informed decisions about loan approvals, pricing, and risk management.

By conducting exploratory data analysis on the loan data, you aim to gain a deeper understanding of the risk and return associated with the business' loans.

Ultimately, your goal is to improve the performance and profitability of the loan portfolio.

## Table of Contents
- [Dataset Schema](#dataset-schema)
- [Scripts](#scripts)
- [db_utils.py](#db_utilspy)
    - [Import](#imports)
    - [Data Extraction](#data-extraction)
    - [Data Type List](#data-type-list)
    - [Data Type Conversion](#data-type-conversion)
    - [DataFrame Overview](#dataframe-overview)
    - [Plotting The Data](#plotting-the-data)
    - [Transforming The Data](#transforming-the-data)
    - [Code Implementation](#code-implementation)
- [db_query.py](#db_querypy)
    - [Imports](#imports-1)
    - [Task 1](#task-1)
    - [Task 2](#task-2)
    - [Task 3](#task-3)
    - [Task 4](#task-4)
    - [Task 5](#task-5)
    - [Code Implementation](#code-implementation-1)
- [License](#license)
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

- **[db_utils.py](#db_utilspy)**: This is the main script, where the initial EDA happens. Here the data is being extracted from the AWS RDS database and loaded into a Pandas dataframe for further analysis. It contains functions which clean and preprocess the data, ensuring correct formatting for the further analysis. This script also identifies if there are any missing values or skewness present amongst the data which may throw off the subsequent plots and analysis being carried out, and after having identified the problematic columns, it will then handle any missing values and deal with the skewness.

- **[db_query.py](#db_querypy)**: Inside this script is where further querying of the dataset happens. The purpose of this script is to answer any questions management has and dive deeper into the dataset to identify any patterns or trends not visible by the previous analysis carried out. After gaining these insights, the company will stand on much better ground to make better educated decisions on which loans to target and which pose huge risks on the company.

## db_utils.py

### Imports
Here are a list of the imports necessary for the db_utils.py script:

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

### Data Extraction
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

### Data Type List
The dataset that was provided by the cloud had numerous columns that were the wrong datatypes or formats. For reference here is a list of what these were originally, and the target datatyoe we need them to be:

| Column Name                       | Original Data Type      | Target Data Type           |
|-----------------------------------|------------------------|---------------------------|
| id                                | int64                  | object                    |
| member_id                         | int64                  | object                    |
| loan_amount                       | int64                  | float64                   |
| funded_amount                     | float64                | float64                   |
| funded_amount_inv                 | float64                | float64                   |
| term                              | object                 | category                  |
| int_rate                          | float64                | float64                   |
| instalment                        | float64                | float64                   |
| grade                             | object                 | category                  |
| sub_grade                         | object                 | category                  |
| employment_length                 | object                 | category                  |
| home_ownership                    | object                 | category                  |
| annual_inc                        | float64                | float64                   |
| verification_status               | object                 | category                  |
| issue_date                        | object                 | datetime64[ns]           |
| loan_status                       | object                 | category                  |
| payment_plan                      | object                 | category                  |
| purpose                           | object                 | category                  |
| dti                               | float64                | float64                   |
| delinq_2yrs                       | int64                  | category                  |
| earliest_credit_line              | object                 | datetime64[ns]           |
| inq_last_6mths                   | int64                  | category                  |
| mths_since_last_delinq           | float64                | category                  |
| mths_since_last_record           | float64                | category                  |
| open_accounts                     | int64                  | category                  |
| total_accounts                    | int64                  | category                  |
| out_prncp                         | float64                | float64                   |
| out_prncp_inv                     | float64                | float64                   |
| total_payment                     | float64                | float64                   |
| total_payment_inv                 | float64                | float64                   |
| total_rec_prncp                  | float64                | float64                   |
| total_rec_int                     | float64                | float64                   |
| total_rec_late_fee                | float64                | float64                   |
| recoveries                        | float64                | float64                   |
| collection_recovery_fee           | float64                | float64                   |
| last_payment_date                 | object                 | datetime64[ns]           |
| last_payment_amount               | float64                | float64                   |
| next_payment_date                 | object                 | datetime64[ns]           |
| last_credit_pull_date             | object                 | datetime64[ns]           |
| collections_12_mths_ex_med       | float64                | category                  |
| mths_since_last_major_derog      | float64                | category                  |
| policy_code                       | int64                  | category                  |
| application_type                  | object                 | category                  |

### Data Type Conversion
Here is where the data type conversion happened. As you can see in the above table, there were a lot of columns in this dataset that were of a dtype that is not convenient for us in the scope of this project. With the skewed columns they operated on numerical dtypes and if a colummn that stated values like how long till someone had their last infraction ended up skewed it would not be of much use to as as the original face values in that column would've been. Many columns which had a set amount of entries were converted to categorical.

```py
class DataTransform:
    def __init__(self, df):
        self.df = df
    
    def convert_to_categorical(self, df):
        # Changing application type
        df['application_type'] = df['application_type'].astype('category')
        # Changing grade type
        df['grade'] = df['grade'].astype('category')
        # Changing home ownership type
        df['home_ownership'] = df['home_ownership'].astype('category')
        # Changing loan status type
        df['loan_status'] = df['loan_status'].astype('category')
        # Changing payment plan type
        df['payment_plan'] = df['payment_plan'].astype('category')
        # Changing purpose type
        df['purpose'] = df['purpose'].astype('category')
        # Changing sub_grade type
        df['sub_grade'] = df['sub_grade'].astype('category')
        # Changing verification status type
        df['verification_status'] = df['verification_status'].astype('category')
        # Changing collections_12_mths_ex_med type
        df['collections_12_mths_ex_med'] = df['collections_12_mths_ex_med'].astype('category')
        # Changing delinq_2yrs
        df['delinq_2yrs'] = df['delinq_2yrs'].astype('category')
        # Changing inq_last_6mths
        df['inq_last_6mths'] = df['inq_last_6mths'].astype('category')
        # Changing open_accounts
        df['open_accounts'] = df['open_accounts'].astype('category')
        # Changing mths_since_last_delinq
        df['mths_since_last_delinq'] = df['mths_since_last_delinq'].astype('category')
        # Changing mths_since_last_record
        df['mths_since_last_record'] = df['mths_since_last_record'].astype('category')
        # Changing total_accounts
        df['total_accounts'] = df['total_accounts'].astype('category')
        # Changing mths_since_last_major_derog
        df['mths_since_last_major_derog'] = df['mths_since_last_major_derog'].astype('category')
        # Changing policy_code
        df['policy_code'] = df['policy_code'].astype('category')

        # Remove the word 'year' and 'years' from all employment_length entries
        df['employment_length'] = df['employment_length'].str.replace(' years', '').str.replace(' year', '').str.strip()
        # Change dtype to category
        df['employment_length'] = df['employment_length'].astype('category')
        # Change the column title to 'employment_length (years)'
        df.rename(columns={'employment_length': 'employment_length (years)'}, inplace=True)

        #Changing term to categorical
        df['term'] = df['term'].str.replace(' months', '', regex=False)
        df.rename(columns={'term': 'term (months)'}, inplace=True)
        # Keep NaN and convert to float64
        df['term (months)'] = df['term (months)'].astype('category')

        return df


    def convert_to_datetime(self, df):
        # Convert to datetime and handle errors
        date_format = '%b-%Y'

        df['earliest_credit_line'] = pd.to_datetime(df['earliest_credit_line'], format=date_format, errors='coerce')
        df['issue_date'] = pd.to_datetime(df['issue_date'], format=date_format, errors='coerce')
        df['last_credit_pull_date'] = pd.to_datetime(df['last_credit_pull_date'], format=date_format, errors='coerce')
        df['last_payment_date'] = pd.to_datetime(df['last_payment_date'], format=date_format, errors='coerce')
        df['next_payment_date'] = pd.to_datetime(df['next_payment_date'], format=date_format, errors='coerce')

        return df
    
    def convert_to_float(self, df):
         # Convert 'loan_amount' to numeric, handling errors
        df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce')
        
        # Ensure the column is in float64 format
        df['loan_amount'] = df['loan_amount'].astype('float64')

        return df
    
    def convert_to_object(self, df):
        df['id'] = df['id'].astype('object')
        df['member_id'] = df['member_id'].astype('object')

        return df
```

### Dataframe Overview

```py
class DataFrameInfo:
    def __init__(self, df):
        self.df = df

    def summarize(self):
        summary = {}

        # Describe all columns
        summary['data_types'] = self.df.dtypes.to_dict()

        # Statistical values
        summary['statistics'] = {
            'mean': self.df.mean(numeric_only=True).to_dict(),
            'median': self.df.median(numeric_only=True).to_dict(),
            'std_dev': self.df.std(numeric_only=True).to_dict(),
        }

        # Count distinct values in categorical columns
        summary['distinct_counts'] = {col: self.df[col].nunique() 
                                      for col in self.df.select_dtypes(['category']).columns}

        # Shape of the DataFrame
        summary['shape'] = self.df.shape

        # Count and percentage of NULL values
        null_counts = self.df.isnull().sum()
        null_percentage = (null_counts / len(self.df)) * 100
        summary['null_values'] = pd.DataFrame({'Count': null_counts, 'Percentage': null_percentage}).to_dict()

        # List of columns
        summary['columns'] = self.df.columns.tolist()

        return summary
```

### Plotting the Data
Here I had the idea to compile all the plots into a PDF file. This was partly due to an error on my part where, at the time, there were 18 columns which had a before and after transformation which I had to sift through manually at the time. So I thought if there was any better way to handle all those plots, and thats when the idea of having everything in one on-hand PDF would be the most convienent addition to this project.

```py
class Plotter:
    def __init__(self):
        pass

    def create_pdf(self, file_path):
        self.pdf = PdfPages(file_path)
    
    def save_plot_to_pdf(self, figure):
        if self.pdf is not None:
            self.pdf.savefig(figure)

    def close_pdf(self):
        if self.pdf is not None:
            self.pdf.close()
    
    def plot_missing_values(self, df):
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        self.save_plot_to_pdf(plt.gcf())  # Save the current figure to PDF
        plt.show()
        plt.close()

    def plot_histogram(self, df, column, title=None):
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        if title:
            plt.title(title)
        else:
            plt.title(f'Histogram of {column}')
        self.save_plot_to_pdf(plt.gcf())  # Save the current figure to PDF
        plt.show()
        plt.close()

    def plot_skewed_columns(self, df, skew_threshold=1.0):
        skewed_columns = df.select_dtypes(include=['float64', 'int64']).skew().sort_values(ascending=False)
        skewed_columns = skewed_columns[skewed_columns > skew_threshold]

        for column in skewed_columns.index:
            print(f"Column: {column}, Skewness: {skewed_columns[column]}")
            self.plot_histogram(df, column)
            plt.show()
            plt.close()
     
        plt.figure(figsize=(10, 6))
        sns.barplot(x=skewed_columns.index, y=skewed_columns.values)
        plt.title('Skewness of Columns')
        plt.xlabel('Columns')
        plt.ylabel('Skewness')
        self.save_plot_to_pdf(plt.gcf())  # Save the summary plot to PDF
        plt.show()  # Show the summary plot
        plt.close()

        return skewed_columns
```

### Transforming the Data
This class is used to clean up the dataset and make the data useable for EDA. One of the problems with datasets is they tend to have missing values. This code accounts for that by providing each column with two options. If columns have more than 20% of their entries missing, they are deemed useless as imputing values can throw off the column entirely. Imputation is the other route this code goes down where if they have less than 20% missing, it is reasonable enough to impute them in. Then skewed transformations are performed, whether it be by logarithmic or rooting methods, and then EDA can be performed on this transformed data.

```py
class DataFrameTransform:
    def __init__(self, df):
        self.df = df

    def check_missing_values(self):
        missing_percent = self.df.isnull().mean() * 100
        return missing_percent[missing_percent > 0]

    def drop_missing_columns(self, threshold=20):
        missing_percent = self.df.isnull().mean() * 100
        columns_to_drop = missing_percent[missing_percent > threshold].index
        self.df.drop(columns=columns_to_drop, inplace=True)

    def impute_missing_values(self):
        for column in self.df.columns:
            if self.df[column].isnull().any():
                if self.df[column].dtype in ['int64', 'float64']:  # Numeric types
                    self.df[column] = self.df[column].fillna(self.df[column].median())
                elif self.df[column].dtype == 'category':  # Categorical types
                    self.df[column] = self.df[column].fillna(self.df[column].mode()[0])
        return self.df
    
    def identify_skewed_columns(self, threshold=1.0):
        skewed_columns = self.df.select_dtypes(include=['float64', 'int64']).skew().sort_values(ascending=False)
        return skewed_columns[skewed_columns.abs() > threshold].index

    def transform_skewed_columns(self, skew_threshold=0.75):
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64'])
        skewed_columns = numeric_columns.skew().sort_values(ascending=False)
        skewed_columns = skewed_columns[skewed_columns.abs() > skew_threshold]

        # Display skewed columns 
        if len(skewed_columns) == 0:
            print("No skewed columns exceed the threshold.")
            return self.df
        
        existing_columns = skewed_columns.index.intersection(self.df.columns)

        print(f"Skewed columns after filtering: {skewed_columns.index.tolist()}")
        print(f"Existing columns in DataFrame: {self.df.columns.tolist()}")

        for column in existing_columns:
            print(f"Transforming {column} with skewness: {skewed_columns[column]}")

            if self.df[column].isnull().any():
                print(f"Warning: Column {column} contains NaN values and will be skipped.")
                continue

            # Apply transformations based on the presence of zero or negative values
            if self.df[column].min() > 0:  # Apply log transformation if no zero/negative values
                self.df[column] = np.log1p(self.df[column])  # Use log1p for numerical stability
            else:
                # Check for negative values
                if (self.df[column] < 0).any():
                    print(f"Warning: Column {column} contains negative values; applying square root transformation.")
                    self.df[column] = np.sqrt(self.df[column] - self.df[column].min() + 1)  # Shift values to make them non-negative
                else:
                    self.df[column] = np.sqrt(self.df[column])  # Apply square root

            print(f"Transformation complete for {column}.")

        return self.df
    
    def run_data_transformation_pipeline(self):
        print("Checking for missing values...")
        missing_values = self.check_missing_values()
        print(f"Missing Values:\n{missing_values}\n")

        print("Dropping columns with >50% missing values...")
        self.drop_missing_columns()

        print("Imputation of missing values...")
        self.impute_missing_values()
        
        print("Identifying skewed columns...")
        skewed_columns = self.identify_skewed_columns()
        print(f"Skewed Columns: {skewed_columns}\n")

        return skewed_columns
```

### Code Implementation
This is where the code is run from and below are all the processes that happen.

```py
if __name__ == "__main__":
    creds = load_creds('credentials.yaml')

    db_connector = RDSDatabaseConnector(**creds)

    db_connector.initialize_engine()

    data_frame = db_connector.extract_data()

    data_transformer = DataTransform(data_frame)

    data_frame = data_transformer.convert_to_categorical(data_frame)
    data_frame = data_transformer.convert_to_datetime(data_frame)
    data_frame = data_transformer.convert_to_float(data_frame)
    data_frame = data_transformer.convert_to_object(data_frame)

    df_info = DataFrameInfo(data_frame)

    summary = df_info.summarize()
    for key, value in summary.items():
        print(f"{key}:\n{value}\n")

    df_transformer = DataFrameTransform(data_frame)

    skewed_columns = df_transformer.run_data_transformation_pipeline()

    missing_values_after = df_transformer.check_missing_values()
    print("Missing Values After Imputation:\n", missing_values_after)

    pretransformed_file_path = os.path.join('Source_Files', 'loan_payments_data_pretransformed.csv')
    db_connector.save_data(data_frame, pretransformed_file_path)

    plotter = Plotter()
    plotter.create_pdf("visualisations.pdf")
    plotter.plot_missing_values(data_frame)

    if not skewed_columns.empty:
        for column in skewed_columns:
            plotter.plot_histogram(data_frame, column, title=f"Before Transformation: {column}")

        data_frame = df_transformer.transform_skewed_columns()  # Call the method once here

        for column in skewed_columns:
            plotter.plot_histogram(data_frame, column, title=f"After Transformation: {column}")

    plotter.close_pdf()

    transformed_file_path = os.path.join('Source_Files', 'loan_payments_data_transformed.csv')
    db_connector.save_data(data_frame, transformed_file_path)

    # Load the saved CSV into a DataFrame
    data_frame = db_connector.load_csv_to_dataframe(transformed_file_path)

```
## db_query.py

### Imports
Here are the imports necessary for the querying script:
```py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
```
### Task 1
Task Objectives: 
- Summarise what percentage of the loans have been currently recovered compared to the total amount to 
    be paid over the loans term including interest.
- Additionally calculate how much will be paid back in 6 months time with interest. 
- Visualise your results on an appropriate graph.

```py
class PaymentStateQuery:

    def __init__(self, loan_data):
        self.loan_data = loan_data

    # Calculate total amount due (loan amount + interest)
    def calculate_total_amount_due(self):
        self.loan_data['total_amount_due'] = self.loan_data['loan_amount'] + (
            self.loan_data['loan_amount'] * self.loan_data['int_rate'] / 100)

    # Calculate percentage of loans recovered
    def calculate_percentage_recovered(self):
        total_recovered = self.loan_data['recoveries'].sum()
        total_due = self.loan_data['total_amount_due'].sum()
        percentage_recovered = (total_recovered / total_due) * 100
        print(f"Percentage of loans recovered: {percentage_recovered:.2f}%")
        return total_recovered, total_due, percentage_recovered

    # Calculate total amount to be paid back in 6 months
    def calculate_payment_in_6_months(self):
        self.loan_data['term (months)'] = self.loan_data['term (months)'].astype(int)  # Convert term to int
        self.loan_data['monthly_payment'] = self.loan_data['total_amount_due'] / self.loan_data['term (months)']
        self.loan_data['payment_in_6_months'] = self.loan_data['monthly_payment'] * 6
        total_payment_in_6_months = self.loan_data['payment_in_6_months'].sum()
        print(f"Total payment to be paid back in 6 months: {total_payment_in_6_months:.2f}")
        return total_payment_in_6_months

    # Plot recovered vs total amount to be paid over the loans term including interest
    def plot_recovered_vs_due(self, total_recovered, total_due, percentage_recovered):
        labels = ['Total Recovered', 'Total Amount Due']
        values = [total_recovered, total_due]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color=['green', 'red'])
        plt.title('Total Recovered vs Total Amount Due')
        plt.ylabel('Amount ($)')
        plt.xlabel('Categories')
        plt.grid(axis='y')

        # Show the percentage recovered on the plot
        plt.text(0, total_recovered + 1000, f'{percentage_recovered:.2f}%', ha='center')

        # Display the plot
        plt.show()
```
### Task 2
Task Objectives:
- The company wants to check what percentage of loans have been a loss to the company:
- Loans marked as Charged Off in the loan_status column represent a loss to the company.
- Calculate the percentage of charged off loans and the total amount that was paid towards these loans 
before being charged off.

```py
class LossCalculation:
    def __init__(self, loan_data):
        self.loan_data = loan_data

    def calculate_loss_percentage(self):
        # Filter loans that are charged off
        charged_off_loans = self.loan_data[self.loan_data['loan_status'] == 'Charged Off']
        
        # Calculate the percentage of charged off loans
        total_loans = len(self.loan_data)
        charged_off_percentage = (len(charged_off_loans) / total_loans) * 100

        # Calculate the total amount paid towards charged off loans
        total_paid_towards_charged_off = charged_off_loans['total_payment'].sum()

        return charged_off_percentage, total_paid_towards_charged_off
```

### Task 3
Task Objectives:
- Calculate the expected loss of the loans marked as Charged Off.
- Calculate the loss in revenue these loans would have generated for the company if they had finished 
- their term. Visualise the loss projected over the remaining term of these loans.

```py
class ExpectedLoss:
    def __init__(self, loan_data):
        self.loan_data = loan_data

    def calculate_expected_loss(self):
        # Filter loans that are Charged Off
        charged_off_loans = self.loan_data[self.loan_data['loan_status'] == 'Charged Off'].copy()

        # Calculate the remaining term
        charged_off_loans['term (months)'] = charged_off_loans['term (months)'].astype(int)  # Convert term to int
        charged_off_loans.loc[:, 'remaining_term'] = (
            charged_off_loans['term (months)'] - (charged_off_loans['total_payment'] / charged_off_loans['instalment'])
        ).clip(lower=0)  # Ensure no negative remaining term

        # Calculate the projected loss
        charged_off_loans.loc[:, 'expected_loss'] = charged_off_loans['remaining_term'] * charged_off_loans['instalment']

        # Total expected loss
        total_expected_loss = charged_off_loans['expected_loss'].sum()
        
        # Calculate total revenue loss these loans would have generated
        charged_off_loans['total_revenue'] = charged_off_loans['remaining_term'] * charged_off_loans['instalment']
        total_revenue_loss = charged_off_loans['total_revenue'].sum()

        print(f"Total Expected Loss from Charged Off Loans: ${total_expected_loss:.2f}")
        print(f"Total Revenue Loss from Charged Off Loans: ${total_revenue_loss:.2f}")

        return total_expected_loss, total_revenue_loss

    def visualize_expected_loss(self):
        total_expected_loss, total_revenue_loss = self.calculate_expected_loss()

        # Plot total expected loss
        plt.figure(figsize=(8, 6))
        plt.bar(['Expected Loss', 'Revenue Loss'], [total_expected_loss, total_revenue_loss], color=['red', 'orange'])
        plt.title("Projected Losses from Charged Off Loans")
        plt.ylabel("Loss Amount ($)")
        plt.xticks(rotation=0)
        plt.grid(axis='y')

        # Show the plot
        plt.tight_layout()  # Improve spacing
        plt.show()
```

### Task 4

Task Objectives:
- There are customers who are currently behind with their loan payments. This subset of customers 
represent a risk to company revenue.
- What percentage do users in this bracket currently represent?
- Calculate the total amount of customers in this bracket and how much loss the company would incur if 
their status was changed to Charged Off.
- What is the projected loss of these loans if the customer were to finish the full loan term?
- If customers that are late on payments converted to Charged Off, what percentage of total expected 
revenue do these customers and the customers who have already defaulted on their loan represent?

```py
class PossibleLoss:
    def __init__(self, loan_data):
        self.loan_data = loan_data

    def calculate_possible_loss(self):
        # Filter customers who are behind on their payments
        late_customers = self.loan_data[self.loan_data['loan_status'].str.contains('Late')].copy()

        # Calculate the total number of late customers
        total_late_customers = late_customers.shape[0]

        # Calculate the total amount of loans for late customers
        total_late_amount = late_customers['loan_amount'].sum()

        # Calculate expected loss if they were to be charged off
        late_customers['term (months)'] = late_customers['term (months)'].astype(int)
        late_customers['remaining_term'] = late_customers['term (months)'] - (late_customers['total_payment'] / late_customers['instalment'])
        late_customers['expected_loss'] = late_customers['remaining_term'] * late_customers['instalment']
        total_expected_loss_late = late_customers['expected_loss'].sum()

        # Calculate total expected revenue
        total_revenue = self.loan_data['loan_amount'].sum()

        # Calculate the percentage of late customers compared to total customers
        total_customers = self.loan_data.shape[0]
        percentage_late_customers = (total_late_customers / total_customers) * 100 if total_customers > 0 else 0

        # Calculate expected loss for charged off loans
        charged_off_loans = self.loan_data[self.loan_data['loan_status'] == 'Charged Off'].copy()
        charged_off_loans['term (months)'] = charged_off_loans['term (months)'].astype(int)
        charged_off_loans['remaining_term'] = charged_off_loans['term (months)'] - (charged_off_loans['total_payment'] / charged_off_loans['instalment'])
        charged_off_loans['expected_loss'] = charged_off_loans['remaining_term'] * charged_off_loans['instalment']
        total_charged_off_loss = charged_off_loans['expected_loss'].sum()

        # Calculate the percentage of total expected revenue that late customers and charged off loans represent
        combined_loss = total_expected_loss_late + total_charged_off_loss
        percentage_combined_loss = (combined_loss / total_revenue) * 100 if total_revenue > 0 else 0

        # Print the results
        print(f"Total Late Customers: {total_late_customers}")
        print(f"Total Amount for Late Customers: ${total_late_amount:.2f}")
        print(f"Projected Loss from Late Customers if Charged Off: ${total_expected_loss_late:.2f}")
        print(f"Percentage of Customers Late on Payments: {percentage_late_customers:.2f}%")
        print(f"Percentage of Total Expected Revenue from Late and Charged Off Loans: {percentage_combined_loss:.2f}%")

        return {
            "total_late_customers": total_late_customers,
            "total_late_amount": total_late_amount,
            "total_expected_loss_late": total_expected_loss_late,
            "percentage_late_customers": percentage_late_customers,
            "percentage_combined_loss": percentage_combined_loss
        }
```
### Task 5

Task Objectives:
- In this task, you will be analysing the data to visualise the possible indicators that a customer 
will not be able to pay the loan.
- You will want to compare columns which might be indicators against customers who have already stopped 
paying and customers who are currently behind on payments.
- Here are some example columns that might indicate that a user might not pay the loan:
    - Does the grade of the loan have an effect on customers not paying?
    - Is the purpose for the loan likely to have an effect?
    - Does the home_ownership value contribute to the likelihood a customer won't pay?
- To help identify which columns will be of interest, first create a subset of these users.
- Make the analysis and determine the columns are contributing to loans not being paid off and visualise any interesting indicators.
- Compare these indicators between loans already charged off and loans that could change to charged off to check if these same factors apply to loans that have the potential to change to "Charged Off".

```py
class LossIndicators:
    def __init__(self, loan_data):
        self.loan_data = loan_data

    def analyze_loss_indicators(self):
        charged_off_loans = self.loan_data[self.loan_data['loan_status'] == 'Charged Off'].copy()
        late_customers = self.loan_data[self.loan_data['loan_status'].str.contains('Late')].copy()

        # Combine both groups for analysis
        charged_off_loans.loc[:, 'status'] = 'Charged Off'
        late_customers['status'] = 'Late'
        combined_data = pd.concat([charged_off_loans, late_customers], axis=0)

        # Set up visualizations for each indicator
        indicators = ['grade', 'purpose', 'home_ownership']
        plt.figure(figsize=(15, 10))
        
        for idx, indicator in enumerate(indicators):
            plt.subplot(2, 2, idx + 1)
            sns.countplot(data=combined_data, x=indicator, hue='status', order=combined_data[indicator].value_counts().index)
            plt.title(f'Comparison of {indicator} by Loan Status')
            plt.xticks(rotation=45)
            plt.xlabel(indicator)
            plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()

        # Analyze the impact of each indicator on loan status
        for indicator in indicators:
            if indicator in combined_data.columns:  # Check if the indicator exists
                print(f'\nAnalysis of {indicator}:\n')
                print(combined_data.groupby([indicator, 'status']).size().unstack().fillna(0))

    def run_analysis(self):
        self.analyze_loss_indicators()
```
### Code Implementation

```py
if __name__ == "__main__":

    df = pd.read_csv(os.path.join('Source_Files', 'loan_payments_data_pretransformed.csv'))

    #Instantiating the classes
    loan_analysis = PaymentStateQuery(df)
    loss_calc = LossCalculation(df)
    expected_loss_calculator = ExpectedLoss(df)
    possible_loss_calculator = PossibleLoss(df)
    loss_indicators = LossIndicators(df)

    #Task 1
    loan_analysis.calculate_total_amount_due()
    total_recovered, total_due, percentage_recovered = loan_analysis.calculate_percentage_recovered()
    total_payment_in_6_months = loan_analysis.calculate_payment_in_6_months()
    loan_analysis.plot_recovered_vs_due(total_recovered, total_due, percentage_recovered)

    #Task 2
    percentage, total_paid = loss_calc.calculate_loss_percentage()
    print(f"Percentage of Charged Off Loans: {percentage:.2f}%")
    print(f"Total Amount Paid Towards Charged Off Loans: ${total_paid:.2f}")

    #Task 3
    expected_loss_calculator.visualize_expected_loss()

    #Task 4
    possible_loss_calculator.calculate_possible_loss()

    #Task 5
    loss_indicators.run_analysis()
```

## License
This project is licensed under AICore