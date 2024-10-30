import pandas as pd

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

        summary = {
        'columns': self.df.columns.tolist(),                               # Column names
        'data_types': self.df.dtypes.astype(str).tolist(),                 # Data types of columns
        'mean': self.df.mean(numeric_only=True).reindex(self.df.columns).tolist(),      # Mean values for numeric columns
        'median': self.df.median(numeric_only=True).reindex(self.df.columns).tolist(),  # Median values for numeric columns
        'std_dev': self.df.std(numeric_only=True).reindex(self.df.columns).tolist(),    # Standard deviation for numeric columns
        'unique_values': [self.df[col].nunique() if col in categorical_cols else 'N/A' for col in self.df.columns],  # Unique values only for categorical columns
        'nulls': self.df.isnull().sum().tolist(),                                      # Number of nulls for each column
        'null_percentage': (self.df.isnull().sum() / total_rows * 100).tolist(),        # Null percentage for each column
        'zero_count': (self.df == 0).sum().tolist()
    }

    # Convert to DataFrame and handle missing data (e.g., non-numeric columns for means)
        summary_df = pd.DataFrame(summary).fillna('N/A')

        return summary_df