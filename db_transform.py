import numpy as np
from scipy import stats

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
                    self.df[column] = self.df[column].fillna(self.df[column].median()) # Fill numeric columns with the median
                elif self.df[column].dtype == 'category':  # Categorical types
                    self.df[column] = self.df[column].fillna(self.df[column].mode()[0]) # Fill categorical columns with the mode
        return self.df

    def identify_skewed_columns(self, threshold=1.0):
        skewed_columns = self.df.select_dtypes(include=['float64', 'int64']).skew().sort_values(ascending=False)
        return skewed_columns[skewed_columns.abs() > threshold].index

    
    def remove_outliers_zscore(self, threshold=3):
        # Select only numeric columns for Z-score calculation
        numeric_df = self.df.select_dtypes(include=['float64', 'int64'])
        
        # Calculate Z-scores for numeric columns
        z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
        
        # Identify rows where any z-score exceeds the threshold
        outlier_condition = (z_scores > threshold).any(axis=1)
        
        # Filter the DataFrame to remove outliers
        self.df = self.df[~outlier_condition]  # Keep only the rows without outliers
        
        # Log how many rows were removed
        removed_count = outlier_condition.sum()
        print(f"Z-Score outlier removal: Removed {removed_count} rows.")

    def remove_outliers_iqr(self):
        original_shape = self.df.shape
        # Remove rows where values are outside the IQR bounds
        for column in self.df.select_dtypes(include=['float64', 'int64']):
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            # Define bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Filter out outliers
            before_removal_count = self.df.shape[0]
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            after_removal_count = self.df.shape[0]
            removed_count = before_removal_count - after_removal_count
            if removed_count > 0:
                print(f"Outliers removed from column '{column}' based on IQR. Removed {removed_count} rows.")
            else:
                print(f"No outliers found in column '{column}' based on IQR.")
        
        # Log shape after removal
        print(f"Shape after IQR outlier removal: {self.df.shape} (original shape was {original_shape}).")
    
    def check_mass_zeroes(self, skew_threshold=0.75, zero_threshold=0.95): #Took this out
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64'])
        skewed_columns = numeric_columns.skew().sort_values(ascending=False)
        skewed_columns = skewed_columns[skewed_columns.abs() > skew_threshold]

        mass_zeroes = []
        for column in skewed_columns.index:
            zero_proportion = (self.df[column] == 0).mean()
            if zero_proportion > zero_threshold:
                print(f"Column '{column}' has {zero_proportion:.2%} of values as zero.")
                mass_zeroes.append(column)

        return mass_zeroes


    def transform_skewed_columns(self, skew_threshold=0.75): #Took this out
        numeric_columns = self.df.select_dtypes(include=['float64', 'int64'])
        skewed_columns = numeric_columns.skew().sort_values(ascending=False)
        skewed_columns = skewed_columns[skewed_columns.abs() > skew_threshold]

        if len(skewed_columns) == 0:
            print("No skewed columns exceed the threshold.")
            return self.df
        
        existing_columns = skewed_columns.index.intersection(self.df.columns)

        print(f"Skewed columns after filtering: {skewed_columns.index.tolist()}")

        for column in existing_columns:
            print(f"Transforming {column} with skewness: {skewed_columns[column]}")

            zero_proportion = (self.df[column] == 0).mean()
            if zero_proportion > 0.95:
                print(f"Skipping {column} - {zero_proportion:.2%} of values are zero.")
                continue
            
            if self.df[column].isnull().any():
                print(f"Warning: Column {column} contains NaN values and will be skipped.")
                continue

            # Handle different skewness scenarios
            if (self.df[column] <= 0).any():
                # Apply Yeo-Johnson for data with zero or negative values
                print(f"Applying Yeo-Johnson transformation to {column}")
                self.df[column], _ = stats.yeojohnson(self.df[column])
            elif self.df[column].min() > 0:
                # Apply log transformation to strictly positive values
                print(f"Applying log1p transformation to {column}")
                self.df[column] = np.log1p(self.df[column])
            else:
                print(f"Applying Box-Cox transformation to {column}")
                self.df[column], _ = stats.boxcox(self.df[column] + 1)  # Shift to ensure positive values
            
            # After transformation, recompute skewness to check effect
            new_skewness = self.df[column].skew()
            print(f"New skewness for {column}: {new_skewness}")
            
        return self.df

    def run_data_transformation_pipeline(self, zscore_outlier_removal=True, iqr_outlier_removal=False): #Took this out
        print("Checking for missing values:")
        missing_values = self.check_missing_values()
        print(f"Missing Values:\n{missing_values}\n")
        print(f"Initial DataFrame shape: {self.df.shape}")

        print("Dropping columns with >50% missing values:")
        self.drop_missing_columns()
        print(f"Shape after dropping missing columns: {self.df.shape}")

        print("Imputation of missing values:")
        self.impute_missing_values()
        print(f"Shape after imputation: {self.df.shape}")

        print("Checking for columns with a high proportion of zeros:")
        mass_zeroes = self.check_mass_zeroes()
        print(f"Mass zeroes identified: {mass_zeroes}")

        if zscore_outlier_removal:
            print("Removing outliers using Z-score:")
            self.remove_outliers_zscore()
            print(f"Shape after Z-score outlier removal: {self.df.shape}")

        if iqr_outlier_removal:
            print("Removing outliers using IQR:")
            self.remove_outliers_iqr()
            print(f"Shape after IQR outlier removal: {self.df.shape}")
        
        print("Identifying skewed columns:")
        skewed_columns = self.identify_skewed_columns()
        print(f"Skewed Columns: {skewed_columns}\n")

        return skewed_columns, mass_zeroes
    