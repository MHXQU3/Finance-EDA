import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

class Plotter:
    '''
    A plotter class that solely carries the responsibility for all the visuals in this script. As there were
    many visualisations being generated I thought it would be a good idea to look up how to save all of them
    into a singular PDF file so one can go back and look at them all and compare without having to iteratively, 
    save them manually. As well as the pre-skewed and post-skewed plots there is also a plot for the missing values. 
    '''
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
    
    def plot_skewed_columns(self, df, skew_threshold=1.0):
        skewed_columns = df.select_dtypes(include=['float64', 'int64']).skew().sort_values(ascending=False)
        skewed_columns = skewed_columns[skewed_columns > skew_threshold]

        for column in skewed_columns.index:
            print(f"Column: {column}, Skewness: {skewed_columns[column]}")
            self.plot_histogram(df, column)  # Plot histogram for each skewed column
        
        # Plot skewness summary
        plt.figure(figsize=(10, 6))
        sns.barplot(x=skewed_columns.index, y=skewed_columns.values)
        plt.title('Skewness of Columns')
        plt.xlabel('Columns')
        plt.ylabel('Skewness')
        self.save_plot_to_pdf(plt.gcf())  # Save the summary plot to PDF
        plt.close()  # Close the figure after saving

        return skewed_columns