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

class DataTransform:
    '''
    This class changes the data types of columns that based on initial checks were not of the correct dtype.
    These dtypes are based on how they would then eventually go on to be used later on in the project, as although
    some columns like months may be better as integers, in the scope of this project as it will be taken as face value
    and not part of arithmetic functions, I believed it would be better off as categorical, as with many of the 
    columns I have changed. 
    '''
    def __init__(self, df):
        self.df = df
    
    def convert_to_categorical(self, df):

        df['application_type'] = df['application_type'].astype('category')
        df['grade'] = df['grade'].astype('category')
        df['home_ownership'] = df['home_ownership'].astype('category')
        df['loan_status'] = df['loan_status'].astype('category')
        df['payment_plan'] = df['payment_plan'].astype('category')
        df['purpose'] = df['purpose'].astype('category')
        df['sub_grade'] = df['sub_grade'].astype('category')
        df['verification_status'] = df['verification_status'].astype('category')
        df['collections_12_mths_ex_med'] = df['collections_12_mths_ex_med'].astype('category')
        df['delinq_2yrs'] = df['delinq_2yrs'].astype('category')
        df['inq_last_6mths'] = df['inq_last_6mths'].astype('category')
        df['open_accounts'] = df['open_accounts'].astype('category')
        df['mths_since_last_delinq'] = df['mths_since_last_delinq'].astype('category')
        df['mths_since_last_record'] = df['mths_since_last_record'].astype('category')
        df['total_accounts'] = df['total_accounts'].astype('category')
        df['mths_since_last_major_derog'] = df['mths_since_last_major_derog'].astype('category')
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
        df['loan_amount'] = df['loan_amount'].astype('float64')

        return df
    
    def convert_to_object(self, df):
        df['id'] = df['id'].astype('object')
        df['member_id'] = df['member_id'].astype('object')

        return df
    
class DataFrameInfo:
    '''
    This class is used for summarising information about the dataframe, which is useful for providing a quick glance
    and overview that will give us an idea of how to proceed with the EDA. This includes numerous statistical
    identifiers and values.
    '''
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
    '''
    A plotter class that solely carries the responsibility for all the visuals in this script. As there were
    many visualisations being generated I thought it would be a good idea to look up how to save all of them
    into a singular PDF file so one can go back and look at them all and compare without having to iteratively, 
    save them manually. As well as the pre-skewed and post-skewed plots there is also a plot for the missing values. 
    '''
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
        sns.heatmap(df.isnull(), cbar=False, cmap='plasma')
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


class DataFrameTransform:
    """
    This class is responsible for transforming and cleaning the pandas DataFrame. It includes methods to check for 
    and report missing values, drop columns with a high percentage of missing data, impute missing values 
    in numeric and categorical columns, identify skewed numeric columns based on a specified threshold, and 
    transform skewed columns to reduce skewness using log or square root transformations.
    """
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
                    print(f"Warning: Column {column} contains negative values")
                    self.df[column] = np.sqrt(self.df[column] - self.df[column].min() + 1)  # Shift values to make them non-negative
                else:
                    self.df[column] = np.sqrt(self.df[column])  # Apply square root

            print(f"Transformation complete for {column}.")

        return self.df
    
    def run_data_transformation_pipeline(self):
        print("Checking for missing values:")
        missing_values = self.check_missing_values()
        print(f"Missing Values:\n{missing_values}\n")

        print("Dropping columns with >50% missing values:")
        self.drop_missing_columns()

        print("Imputation of missing values:")
        self.impute_missing_values()
        
        print("Identifying skewed columns:")
        skewed_columns = self.identify_skewed_columns()
        print(f"Skewed Columns: {skewed_columns}\n")

        return skewed_columns



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

        data_frame = df_transformer.transform_skewed_columns() 

        for column in skewed_columns:
            plotter.plot_histogram(data_frame, column, title=f"After Transformation: {column}")

    plotter.close_pdf()

    transformed_file_path = os.path.join('Source_Files', 'loan_payments_data_transformed.csv')
    db_connector.save_data(data_frame, transformed_file_path)

    data_frame = db_connector.load_csv_to_dataframe(transformed_file_path)

    if data_frame is not None:
        print(data_frame.shape)  # Display the shape of the DataFrame