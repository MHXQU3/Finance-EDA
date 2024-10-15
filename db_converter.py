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
        self.cat_cols = ['application_type', 'grade', 'home_ownership', 'loan_status', 'payment_plan', 'purpose', 'sub_grade', 'verification_status', 'collections_12_mths_ex_med', 'delinq_2yrs', 'inq_last_6mths', 'open_accounts', 'mths_since_last_delinq', 'mths_since_last_record', 'total_accounts', 'mths_since_last_major_derog', 'policy_code', 'employment_length', 'term']
        self.date_cols = ['earliest_credit_line', 'issue_date', 'last_credit_pull_date', 'last_payment_date', 'next_payment_date']
        self.float_cols = ['loan_amount']
        self.object_cols = ['id', 'member_id']

    def convert_to_categorical(self):
        
        # Convert all general categorical columns
        for col in self.cat_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
            else:
                print(f"Warning: Column '{col}' not found in DataFrame.")

        # Handle columns with 'years' or 'months' in the values
        for i in range(len(self.cat_cols)):
            col = self.cat_cols[i]

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
                self.cat_cols[i] = f'{col} (years)'

            # Check if 'months2' appears in any entry of the column
            elif self.df[col].astype(str).str.contains('months').any():
                self.df[col] = self.df[col].str.replace(' months', '').str.strip()
                self.df[col] = self.df[col].astype('category')
                # Rename the column in the DataFrame
                self.df.rename(columns={col: f'{col} (months)'}, inplace=True)
                # Update the column name in the list
                self.cat_cols[i] = f'{col} (months)'

        return self.df

        
    
    
                

    def convert_to_datetime(self):
        for col in self.date_cols:
            self.df[col] = pd.to_datetime(self.df[col], format= '%b-%Y', errors='coerce')
        return self.df

    def convert_to_float(self):
        for col in self.float_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')  # Convert to numeric, coercing errors
            self.df[col] = self.df[col].astype('float64')  # Ensure the type is float64
        return self.df

    def convert_to_object(self):
        for col in self.object_cols:
            self.df[col] = self.df[col].astype('object')
        return self.df