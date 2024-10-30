import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox, skew, yeojohnson

class DataFrameTransform:
    def __init__(self, df):
        self.df = df

    def impute_missing_values(self):
        for column in self.df.columns:
            if self.df[column].isnull().any():
                if self.df[column].dtype in ['int64', 'float64']:  # Numeric types
                    self.df[column] = self.df[column].fillna(self.df[column].median()) # Fill numeric columns with the median
                elif self.df[column].dtype == 'category':  # Categorical types
                    self.df[column] = self.df[column].fillna(self.df[column].mode()[0]) # Fill categorical columns with the mode
        return self.df
    
    def impute_mean_by_category(self, Category_Column, Value_Column):
        """Replaces null value with median value"""
        mean_per_category = self.df.groupby(Category_Column)[Value_Column].transform('mean')  #Calculate mean values per category
        ValueColumn_filled = self.df[Value_Column].fillna(mean_per_category, inplace=True)
        return ValueColumn_filled
    
    def impute_previous_row_value(self, Category_Column, Value_Column):
        """Imputes the value from the previous row into null values in the named value_column, when filtered by category within the category_column"""
        categories = self.df[Category_Column].unique()
        # Loop through each category
        for category in categories:
            # Filter the DataFrame for the current category
            mask = self.df[Category_Column] == category
            # Back fill missing values within the category
            self.df.loc[mask, Value_Column] = self.df.loc[mask, Value_Column].bfill()
        return self.df

    def impute_term(self, value_column, loan_amount, int_rate, instalment):
        """Imputes a loan term into the value_column (which in this project is loan_term) using a calculation that takes values from the loan amount, interest 
        rate and installment amount columns. As a by-product of this process, the annual interest rate is converted from a percentage to a decimal."""
        import math
        import numpy as np
        
        # Ensure the relevant columns are numeric
        self.df[loan_amount] = pd.to_numeric(self.df[loan_amount])
        self.df[int_rate] = pd.to_numeric(self.df[int_rate]) / 100  # Convert annual interest rate to decimal
        self.df[instalment] = pd.to_numeric(self.df[instalment])

        change_count = 0
        
        def calc_loan_term_months(row):
            nonlocal change_count 
            # Only calculate if the existing term is not null
            if pd.isna(row[value_column]):  # If term is NaN, leave it unchanged
                return row
            
            try:
                # Calculate the loan term using the provided formula
                calculated_term = round(-(
                    math.log(1 - ((row[loan_amount] * (row[int_rate] / 12)) / row[instalment])) 
                ) / (math.log(1 + (row[int_rate] / 12))))
                # Check if the calculated term is different from the existing value
                if calculated_term != row[value_column]:
                    row[value_column] = calculated_term  # Update the term with the new value
                    change_count += 1  # Increment the change counter
            except Exception as e:
                print(f"Error processing row ID {row.get('id', 'N/A')}: {e}")
                # Leave the term unchanged in case of an exception
            return row
        
        self.df = self.df.apply(calc_loan_term_months, axis=1)
        print(f"Number of rows changed: {change_count}")
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

    def remove_outliers_iqr(self, column):
        print(f"Initial shape before removing outliers: {self.df.shape}")
        
        # Calculate Q1, Q3, and IQR
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

        # Filter out outliers
        self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]

        print(f"Final shape after removing outliers: {self.df.shape}")

        # Check for remaining outliers
        remaining_outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        if not remaining_outliers.empty:
            print(f"Warning: There are still outliers in column {column}: {remaining_outliers}")
        else:
            print(f"No remaining outliers in column {column}.")
        
        return self.df
    
    def apply_boxcox(self, column):
        print(f"Applying Box-Cox transformation to {column}")
        self.df[column], _ = boxcox(self.df[column])
        return self.df
    
    def apply_yeojohnson(self, column):
        print(f"Applying Yeo-Johnson transformation to {column}")
        self.df[column], _ = yeojohnson(self.df[column])  
        return self.df
    
    def apply_transformations(self, column, transformation):

        if transformation == 'box_cox':
            print(f"Applying Box-Cox transformation to {column}")
            self.df[column], _ = boxcox(self.df[column])
        
        elif transformation == 'yeo_johnson':
            print(f"Applying Yeo-Johnson transformation to {column}")
            self.df[column], _ = yeojohnson(self.df[column])            
    
        return self.df
