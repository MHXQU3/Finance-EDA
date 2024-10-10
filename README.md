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
- [Import](#imports)
- [Data Extraction](#data-extraction)
- [Data Type List](#data-type-list)
- [Data Type Conversion](#data-type-conversion)
- [DataFrame Overview](#dataframe-overview)
- [Plotting The Data](#plotting-the-data)
- [Transforming The Data](#transforming-the-data)
- [Code Implementation](#code-implementation)

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

## Data Type List
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

## Data Type Conversion

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

## Dataframe Overview

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

## Plotting the Data

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

## Transforming the Data

```py
class DataFrameTransform:
    def __init__(self, df):
        self.df = df

    def check_missing_values(self):
        missing_percent = self.df.isnull().mean() * 100
        return missing_percent[missing_percent > 0]

    def drop_missing_columns(self, threshold=50):
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

## Code Implementation

```py
if __name__ == "__main__":
    # Load credentials
    creds = load_creds('credentials.yaml')

    # Create an instance of the RDSDatabaseConnector using dictionary unpacking
    db_connector = RDSDatabaseConnector(**creds)

    # Initialize the database engine
    db_connector.initialize_engine()

    # Extract data from the loan_payments table
    data_frame = db_connector.extract_data()

    # Create an instance of DataTransform
    data_transformer = DataTransform(data_frame)

    # Convert columns to categorical types
    data_frame = data_transformer.convert_to_categorical(data_frame)

    # Convert date columns to datetime and format
    data_frame = data_transformer.convert_to_datetime(data_frame)

    # Convert column to float and format
    data_frame = data_transformer.convert_to_float(data_frame)

    # Convert column to object and format
    data_frame = data_transformer.convert_to_object(data_frame)

    # Create an instance of DataFrameInfo
    df_info = DataFrameInfo(data_frame)

    # Get and print the summary
    summary = df_info.summarize()
    for key, value in summary.items():
        print(f"{key}:\n{value}\n")

     # Create an instance of DataFrameTransform for handling missing values
    df_transformer = DataFrameTransform(data_frame)

   # Run the data transformation pipeline
    skewed_columns = df_transformer.run_data_transformation_pipeline()

    # Check for missing values after transformation
    missing_values_after = df_transformer.check_missing_values()
    print("Missing Values After Imputation:\n", missing_values_after)

    # Save the pre-transformed data to a CSV for reference
    pretransformed_file_path = os.path.join('Source_Files', 'loan_payments_data_pretransformed.csv')
    db_connector.save_data(data_frame, pretransformed_file_path)

    # Create an instance of Plotter to visualize missing values
    plotter = Plotter()
    plotter.create_pdf("visualisations.pdf")
    plotter.plot_missing_values(data_frame)

    # Visualize skewed columns before transformation
    if not skewed_columns.empty:
        for column in skewed_columns:
            plotter.plot_histogram(data_frame, column, title=f"Before Transformation: {column}")

        # Apply transformations to reduce skewness
        data_frame = df_transformer.transform_skewed_columns()  # Call the method once here

        # Visualize skewed columns after transformation
        for column in skewed_columns:
            plotter.plot_histogram(data_frame, column, title=f"After Transformation: {column}")

    plotter.close_pdf()

    transformed_file_path = os.path.join('Source_Files', 'loan_payments_data_transformed.csv')
    db_connector.save_data(data_frame, transformed_file_path)

    # Load the saved CSV into a DataFrame
    data_frame = db_connector.load_csv_to_dataframe(transformed_file_path)

```