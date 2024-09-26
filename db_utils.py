import pandas as pd
#import numpy as np
import yaml
from sqlalchemy import create_engine, inspect
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


    import pandas as pd

def load_csv_to_dataframe(file_path):
    
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




if __name__ == "__main__":
    # Load credentials
    creds = load_creds('credentials.yaml')

    # Create an instance of the RDSDatabaseConnector using dictionary unpacking
    db_connector = RDSDatabaseConnector(**creds)

    # Initialize the database engine
    db_connector.initialize_engine()

    # Extract data from the loan_payments table
    data_frame = db_connector.extract_data()

    # Save the data to a CSV file
    db_connector.save_data(data_frame, 'loan_payments_data.csv')

    file_path = 'loan_payments_data.csv'
    data_frame = load_csv_to_dataframe(file_path)

    if data_frame is not None:
        print(data_frame.shape)  # Display the shape of the DataFrame
