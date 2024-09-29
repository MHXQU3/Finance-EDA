import pandas as pd
import numpy as np
import yaml
from sqlalchemy import create_engine, inspect
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import os

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

        # Remove the word 'year' and 'years' from all employment_length entries
        df['employment_length'] = df['employment_length'].str.replace(' years', '').str.replace(' year', '').str.strip()
        # Change dtype to category
        df['employment_length'] = df['employment_length'].astype('category')
        # Change the column title to 'employment_length (years)'
        df.rename(columns={'employment_length': 'employment_length (years)'}, inplace=True)

        return df

    def convert_to_datetime(self, df):
        # Convert to datetime and handle errors
        df['earliest_credit_line'] = pd.to_datetime(df['earliest_credit_line'], errors='coerce')
        df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')
        df['last_credit_pull_date'] = pd.to_datetime(df['last_credit_pull_date'], errors='coerce')
        df['last_payment_date'] = pd.to_datetime(df['last_payment_date'], errors='coerce')
        df['next_payment_date'] = pd.to_datetime(df['next_payment_date'], errors='coerce')

        # Format to year-month
        df['earliest_credit_line'] = df['earliest_credit_line'].dt.strftime('%Y-%m')
        df['issue_date'] = df['issue_date'].dt.strftime('%Y-%m')
        df['last_credit_pull_date'] = df['last_credit_pull_date'].dt.strftime('%Y-%m')
        df['last_payment_date'] = df['last_payment_date'].dt.strftime('%Y-%m') 
        df['next_payment_date'] = df['next_payment_date'].dt.strftime('%Y-%m')

        return df
    
    def convert_to_float(self, df):
        # Remove ' months', convert to numeric
        df['term'] = pd.to_numeric(df['term'].str.replace(' months', '', regex=False), errors='coerce')
        # Rename the column to 'term (months)'
        df.rename(columns={'term': 'term (months)'}, inplace=True)
        # Keep NaN and convert to float64
        df['term (months)'] = pd.to_numeric(df['term (months)'], errors='coerce')
        

        return df
    
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

class Plotter:
    def __init__(self):
        pass
    
    def plot_missing_values(self, df):
        plt.figure(figsize=(12, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()


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

    # Create an instance of DataFrameInfo
    df_info = DataFrameInfo(data_frame)

    # Get and print the summary
    summary = df_info.summarize()
    for key, value in summary.items():
        print(f"{key}:\n{value}\n")

     # Create an instance of DataFrameTransform for handling missing values
    df_transformer = DataFrameTransform(data_frame)

    # Check for missing values
    missing_values = df_transformer.check_missing_values()
    print("Missing Values Percentage:\n", missing_values)

    # Drop columns with more than 50% missing values
    df_transformer.drop_missing_columns(threshold=50)

    # Impute remaining missing values
    data_frame = df_transformer.impute_missing_values()

    # Run NULL checking again to confirm
    missing_values_after = df_transformer.check_missing_values()
    print("Missing Values After Imputation:\n", missing_values_after)

    # Create an instance of Plotter to visualize missing values
    plotter = Plotter()
    plotter.plot_missing_values(data_frame)

    # Save the data to a CSV file
    db_connector.save_data(data_frame, 'loan_payments_data.csv')

    file_path = 'loan_payments_data.csv'
    data_frame = db_connector.load_csv_to_dataframe(file_path)

    if data_frame is not None:
        print(data_frame.shape)  # Display the shape of the DataFrame
