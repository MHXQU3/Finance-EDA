import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns 
import statsmodels.api as sm

class Plotter:
    def __init__(self, df):
        self.df = df

    def create_pdf(self, file_path):
        self.pdf = PdfPages(file_path)
    
    def save_plot_to_pdf(self, figure):
        if self.pdf is not None:
            self.pdf.savefig(figure)

    def close_pdf(self):
        if self.pdf is not None:
            self.pdf.close()

    def discrete_probability_distribution(self, column_name):
        """Produces a discrete probability distribution for the named column."""
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.rc("axes.spines", top=False, right=False)

        # Calculate value counts and convert to probabilities
        probs = self.df[column_name].value_counts(normalize=True)

        # Create bar plot
        dpd=sns.barplot(y=probs.index, x=probs.values, color='r')

        plt.xlabel('Probability')
        plt.ylabel('Values')
        plt.title('Discrete Probability Distribution')
        plt.show()    

    def histogram_df_columns(self):
        """Produces a histogram for all numeric columns within the dataframe."""

        numeric_columns = self.df.select_dtypes(include=['number'])

        fig, axes = plt.subplots(4, 4, figsize=(15, 15))  
        ax = axes.flatten()

        for i, col in enumerate(numeric_columns.columns):
            sns.histplot(self.df[col], ax=ax[i], kde=True, bins=20, color='blue', alpha=0.5)
            ax[i].set_title(col)
            ax[i].set_xlabel('Value')
            ax[i].set_ylabel('Frequency')

            # Set tick labels to plain style (non-scientific notation)
            if pd.api.types.is_numeric_dtype(numeric_columns[col]):
                ax[i].get_xaxis().get_major_formatter().set_scientific(False)
                ax[i].get_yaxis().get_major_formatter().set_scientific(False)

        plt.tight_layout()
        plt.show()


    def qqplot_df_columns(self):
        """Produces a QQplot for all numeric columns within the dataframe."""

        numeric_columns = self.df.select_dtypes(include=['number'])

        fig, axes = plt.subplots(4, 4, figsize=(15, 15))  
        ax = axes.flatten()

        for i, col in enumerate(numeric_columns.columns):
            sm.qqplot(self.df[col], line='s', ax=ax[i])
            ax[i].set_title(col)
            ax[i].set_xlabel('Theoretical Quantiles')
            ax[i].set_ylabel('Sample Quantiles')
            ax[i].get_xaxis().get_major_formatter().set_scientific(False)
            ax[i].get_yaxis().get_major_formatter().set_scientific(False)

        plt.tight_layout()
        plt.show()

    def boxplot(self):
        """Produces a boxplot for all numeric columns within the dataframe."""

        numeric_columns = self.df.select_dtypes(include=['number'])

        fig, axes = plt.subplots(4, 4, figsize=(50, 40))  
        ax = axes.flatten()

        for i, col in enumerate(numeric_columns.columns):
            sns.boxplot(x=numeric_columns[col], ax=ax[i])
            ax[i].set_title(col)
            ax[i].set_ylabel('Value')

        plt.tight_layout()
        plt.show()

    def heatmap(self, matrix):
        """Produces a heatmap based on the matrix given."""

        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(matrix, annot=False, cmap='PRGn', center=0)
        heatmap.set_title('Correlation Matrix Heatmap', fontdict={'fontsize':18}, pad=16)

        # Display the heatmap
        plt.show()
