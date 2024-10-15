import numpy as np

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
                    self.df[column] = self.df[column].fillna(self.df[column].median()) # Fills these in with the median values to avoid skewing data
                elif self.df[column].dtype == 'category':  # Categorical types
                    self.df[column] = self.df[column].fillna(self.df[column].mode()[0]) # Fills these in with the mode to ensure consistency
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

            # Apply transformations based on the presence of zero or -ve values
            if self.df[column].min() > 0:  # Apply log transformation to non zero/-ve values
                self.df[column] = np.log1p(self.df[column])  # log1p good for numerical stabilty
                # Check for negative values
                if (self.df[column] < 0).any():
                    print(f"Warning: Column {column} contains negative values")
                    self.df[column] = np.sqrt(self.df[column] - self.df[column].min() + 1)  # Shift values to make them non-negative
                else:
                    self.df[column] = np.sqrt(self.df[column]) 

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
