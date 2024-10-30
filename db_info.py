import pandas as pd
import numpy as np
from scipy.stats import boxcox, skew, yeojohnson

class DataFrameInfo:
    '''
    This class is used for summarising information about the dataframe, which is useful for providing a quick glance
    and overview that will give us an idea of how to proceed with the EDA. This includes numerous statistical
    identifiers and values.
    '''
    def __init__(self, df):
        self.df = df

    def summarize(self):

        total_rows = self.df.shape[0]

        # Filter out categorical columns (object or category types)
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = self.df.select_dtypes(include=['number']).columns

        summary = {
        'columns': self.df.columns.tolist(),                               # Column names
        'data_types': self.df.dtypes.astype(str).tolist(),                 # Data types of columns
        'mode': [self.df[col].mode()[0] if col in categorical_cols else 'N/A' for col in self.df.columns],  # Mode for categorical columns
        'mean': self.df.mean(numeric_only=True).reindex(self.df.columns).tolist(),      # Mean values for numeric columns
        'median': self.df.median(numeric_only=True).reindex(self.df.columns).tolist(),  # Median values for numeric columns
        'std_dev': self.df.std(numeric_only=True).reindex(self.df.columns).tolist(),    # Standard deviation for numeric columns
        'unique_values': [self.df[col].nunique() if col in categorical_cols else 'N/A' for col in self.df.columns],  # Unique values only for categorical columns
        'nulls': self.df.isnull().sum().tolist(),                                      # Number of nulls for each column
        'null_percentage': (self.df.isnull().sum() / total_rows * 100).tolist(),        # Null percentage for each column
        'zero_count': (self.df == 0).sum().tolist(),
        'zero_percentage': ((self.df == 0).sum() / total_rows * 100).tolist(),
        'skewness':[self.df[col].skew() if col in numeric_cols else 'N/A' for col in self.df.columns]
    }
        

    # Convert to DataFrame and handle missing data (e.g., non-numeric columns for means)
        summary_df = pd.DataFrame(summary).fillna('N/A')

        return summary_df
    
    def analyze_skew_methods(self, columns):
        """Analyses the skewness of each column, to return a table of data cotaining column name, the original skew, box cox transformation and Yeo 
        Johnson tnsformation"""
        results = []

        for column in columns:
            original_data = self.df[column].dropna()
            
            # Calculate original skewness
            original_skewness = skew(original_data)
            
            # Box-Cox transformation (only works with positive values)
            if (original_data > 0).all():
                boxcox_transformed, _ = boxcox(original_data)
                boxcox_skewness = skew(boxcox_transformed)
            else:
                boxcox_skewness = np.nan
            
            # Log transformation (only works with positive values)
            if (original_data > 0).all():
                log_transformed = np.log1p(original_data)
                log_skewness = skew(log_transformed)
            else:
                log_skewness = np.nan
            
            # Yeo-Johnson transformation (works with all values)
            yeo_transformed, _ = yeojohnson(original_data)
            yeo_skewness = skew(yeo_transformed)
            
            # Append results
            results.append({
                'Column': column,
                'Original Skewness': original_skewness,
                'Box-Cox Skewness': boxcox_skewness,
                'Log Skewness': log_skewness,
                'Yeo-Johnson Skewness': yeo_skewness
            })
    
        return pd.DataFrame(results)
    
    def correlation_columns(self):
        """Returns the correlation between 2 columns to allow production of a correlation heatmap"""
        return self.df.select_dtypes(include=['number']).corr()
