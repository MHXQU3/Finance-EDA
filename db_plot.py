import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy import stats

class Plotter:

    def __init__(self):
        self.pdf = None  # Initialize pdf as None
    
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
        #plt.close()  # Close the figure after saving

    def plot_histogram(self, df, column, title=None):
        if column in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df[column], kde=True)  # Kernel density estimate
            if title:
                plt.title(title)
            else:
                plt.title(f'Histogram of {column}')
            self.save_plot_to_pdf(plt.gcf())  # Save the current figure to PDF
            #plt.close()  # Close the figure after saving
        else:
            print(f"Column '{column}' does not exist in the DataFrame.")
    
    def plot_skewed_columns(self, df, skew_threshold=1.0, mass_zeroes=[]):
        skewed_columns = df.select_dtypes(include=['float64', 'int64']).skew().sort_values(ascending=False)
        skewed_columns = skewed_columns[skewed_columns > skew_threshold]

        existing_skewed_columns = skewed_columns.index.difference(mass_zeroes)

        for column in existing_skewed_columns:
            print(f"Column: {column}, Skewness: {skewed_columns[column]}")
            self.plot_histogram(df, column)  # Plot histogram for each skewed column
        
        # Plot skewness summary
        if len(existing_skewed_columns) > 0:
            plt.figure(figsize=(10, 6))
            sns.barplot(x=existing_skewed_columns, y=skewed_columns[existing_skewed_columns])
            plt.title('Skewness of Columns')
            plt.xlabel('Columns')
            plt.ylabel('Skewness')
            self.save_plot_to_pdf(plt.gcf())  # Save the summary plot to PDF
            plt.close()  # Close the figure after saving
        else:
            print("No skewed columns left to plot.")

        return existing_skewed_columns
    
    def plot_correlation_matrix(self, df):
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=['float64', 'int64'])

        plt.figure(figsize=(10,6))
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar=True)
        plt.title('Correlation Matrix Heatmap')
        self.save_plot_to_pdf(plt.gcf())
        plt.close()  # Close the figure after saving

    def plot_zscore_boxplots(self, df):
        numeric_columns = df.select_dtypes(include=['float64', 'int64'])
        z_scores = np.abs(stats.zscore(numeric_columns, nan_policy='omit'))
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=z_scores)
        plt.title('Box Plot of Z-Scores for Numeric Columns')
        plt.ylabel('Z-Score')
        self.save_plot_to_pdf(plt.gcf())
        plt.close()  # Close the figure after saving