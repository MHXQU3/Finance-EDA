import pandas as pd

class DataDeletion:
    def __init__(self, df):
        self.df = df

    def delete_column(self, column_name):
        """Deletes the named column from the dataframe"""
        drop_df = self.df.drop(column_name, axis=1, inplace=True) # Drops named column
        return drop_df
    
    def delete_row_using_id(self, id_no):
        """Deletes the named row from the dataframe, by matching the value in the id_no column"""
        self.df.drop(self.df[self.df['id'] == id_no].index, inplace=True)
        return self.df
    
    def delete_row(self, column_name):
        """Deletes rows with null values in the specified column from the dataframe"""
        drop_df = self.df.dropna(subset=[column_name], inplace=True) # Drops rows with null values in named column
        return drop_df
    
    def delete_row_if_both_null(self, column_name_1, column_name_2):
        """Deletes rows with null values in both the specified columns from the dataframe"""
        drop_df = self.df.dropna(subset=[column_name_1, column_name_2], how='all', inplace=True) # Drops rows with null values in both named columns
        return drop_df
