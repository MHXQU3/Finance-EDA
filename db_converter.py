import pandas as pd 

class DataConverter:
    '''
    This class changes the data types of columns that based on initial checks were not of the correct dtype.
    These dtypes are based on how they would then eventually go on to be used later on in the project, as although
    some columns like months may be better as integers, in the scope of this project as it will be taken as face value
    and not part of arithmetic functions, I believed it would be better off as categorical, as with many of the 
    columns I have changed. 
    '''
    def __init__(self, df):
        self.df = df

    def convert_to_categorical(self, cat_cols):
        
        # Convert all general categorical columns
        for col in cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
            else:
                print(f"Warning: Column '{col}' not found in DataFrame.")

        # Handle columns with 'years' or 'months' in the values
        for i in range(len(cat_cols)):
            col = cat_cols[i]

                # Skip if the column is not in the DataFrame
            if col not in self.df.columns:
                continue

            # Check if 'years' appears in any entry of the column
            if self.df[col].astype(str).str.contains('year').any():
                self.df[col] = self.df[col].str.replace(' years', '').str.replace(' year', '').str.strip()
                self.df[col] = self.df[col].astype('category')
                # Rename the column in the DataFrame
                self.df.rename(columns={col: f'{col} (years)'}, inplace=True)
                # Update the column name in the list
                cat_cols[i] = f'{col} (years)'

            # Check if 'months2' appears in any entry of the column
            elif self.df[col].astype(str).str.contains('months').any():
                self.df[col] = self.df[col].str.replace(' months', '').str.strip()
                self.df[col] = self.df[col].astype('category')
                # Rename the column in the DataFrame
                self.df.rename(columns={col: f'{col} (months)'}, inplace=True)
                # Update the column name in the list
                cat_cols[i] = f'{col} (months)'

        return self.df

    def convert_to_datetime(self, date_cols):
        for col in date_cols:
            self.df[col] = pd.to_datetime(self.df[col], format= '%b-%Y', errors='coerce')
        return self.df

    def convert_to_float(self, float_cols):
        for col in float_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')  # Convert to numeric, coercing errors
            self.df[col] = self.df[col].astype('float64')  # Ensure the type is float64
        return self.df

    def convert_to_object(self, object_cols):
        for col in object_cols:
            self.df[col] = self.df[col].astype('object')
        return self.df
