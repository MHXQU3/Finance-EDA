import pandas as pd
import numpy as np
import yaml
from sqlalchemy import create_engine, inspect
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import psycopg2
import os

def load_creds(file_path):
    with open(file_path, 'r') as file:
        credentials = yaml.safe_load(file)
    return credentials

class RDSDatabaseConnector:
    '''
    Connects to the RDS PGSQL database based on the details in the credential file. Then extracts data
    from the loan payments table, saving it to a csv file after which it is then loaded into a pandas dataframe. 
    '''
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








# if __name__ == "__main__":
#     creds = load_creds('credentials.yaml')

#     db_connector = RDSDatabaseConnector(**creds)

#     db_connector.initialize_engine()

#     data_frame = db_connector.extract_data()

#     data_transformer = DataTransform(data_frame)

#     data_frame = data_transformer.convert_to_categorical(data_frame)
#     data_frame = data_transformer.convert_to_datetime(data_frame)
#     data_frame = data_transformer.convert_to_float(data_frame)
#     data_frame = data_transformer.convert_to_object(data_frame)

#     df_info = DataFrameInfo(data_frame)

#     summary = df_info.summarize()
#     for key, value in summary.items():
#         print(f"{key}:\n{value}\n")

#     df_transformer = DataFrameTransform(data_frame)

#     skewed_columns = df_transformer.run_data_transformation_pipeline()

#     missing_values_after = df_transformer.check_missing_values()
#     print("Missing Values After Imputation:\n", missing_values_after)

#     pretransformed_file_path = os.path.join('Source_Files', 'loan_payments_data_pretransformed.csv')
#     db_connector.save_data(data_frame, pretransformed_file_path)

#     plotter = Plotter()
#     plotter.create_pdf("visualisations.pdf")
#     plotter.plot_missing_values(data_frame)

#     if not skewed_columns.empty:
#         for column in skewed_columns:
#             plotter.plot_histogram(data_frame, column, title=f"Before Transformation: {column}")

#         data_frame = df_transformer.transform_skewed_columns() 

#         for column in skewed_columns:
#             plotter.plot_histogram(data_frame, column, title=f"After Transformation: {column}")

#     plotter.close_pdf()

#     transformed_file_path = os.path.join('Source_Files', 'loan_payments_data_transformed.csv')
#     db_connector.save_data(data_frame, transformed_file_path)

#     data_frame = db_connector.load_csv_to_dataframe(transformed_file_path)

#     if data_frame is not None:
#         print(data_frame.shape)  