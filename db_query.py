import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PaymentStateQuery: #Task 1
    '''
    - Summarise what percentage of the loans have been currently recovered compared to the total amount to 
    be paid over the loans term including interest.
    - Additionally calculate how much will be paid back in 6 months time with interest. 
    - Visualise your results on an appropriate graph.
    '''
    def __init__(self, loan_data):
        self.loan_data = loan_data

    # Calculate total amount due (loan amount + interest)
    def calculate_total_amount_due(self):
        self.loan_data['total_amount_due'] = self.loan_data['loan_amount'] + (
            self.loan_data['loan_amount'] * self.loan_data['int_rate'] / 100)

    # Calculate percentage of loans recovered
    def calculate_percentage_recovered(self):
        total_recovered = self.loan_data['recoveries'].sum() #total payment
        total_due = self.loan_data['total_amount_due'].sum()
        percentage_recovered = (total_recovered / total_due) * 100
        print(f"Percentage of loans recovered: {percentage_recovered:.2f}%")
        return total_recovered, total_due, percentage_recovered 

    # Calculate total amount to be paid back in 6 months
    def calculate_payment_in_6_months(self):
        self.loan_data['term (months)'] = self.loan_data['term (months)'].astype(int)  # Convert term to int
        self.loan_data['monthly_payment'] = self.loan_data['total_amount_due'] / self.loan_data['term (months)']
        self.loan_data['payment_in_6_months'] = self.loan_data['monthly_payment'] * 6
        total_payment_in_6_months = self.loan_data['payment_in_6_months'].sum()
        print(f"Total payment to be paid back in 6 months: {total_payment_in_6_months:.2f}")
        return total_payment_in_6_months #not all loans will have 6 months left
    # make start date ato get months so far - away from term for amount of months to go
    # if amount of months is 0 do nothing
    # if amount of months is 6 or less x by this number
    # if 6 or higher then times by 6 
    # filter off by all of charged off loans

    # Plot recovered vs total amount to be paid over the loans term including interest
    def plot_recovered_vs_due(self, total_recovered, total_due, percentage_recovered):
        labels = ['Total Recovered', 'Total Amount Due']
        values = [total_recovered, total_due]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, values, color=['green', 'red'])
        plt.title('Total Recovered vs Total Amount Due')
        plt.ylabel('Amount ($)')
        plt.xlabel('Categories')
        plt.grid(axis='y')

        # Show the percentage recovered on the plot
        plt.text(0, total_recovered + 1000, f'{percentage_recovered:.2f}%', ha='center')

        # Display the plot
        plt.show()

class LossCalculation: #Task 2
    '''
    - The company wants to check what percentage of loans have been a loss to the company:
    - Loans marked as Charged Off in the loan_status column represent a loss to the company.
    - Calculate the percentage of charged off loans and the total amount that was paid towards these loans 
    before being charged off.
    '''
    def __init__(self, loan_data):
        self.loan_data = loan_data

    def calculate_loss_percentage(self):
        # Filter loans that are charged off
        charged_off_loans = self.loan_data[self.loan_data['loan_status'] == 'Charged Off']
        
        # Calculate the percentage of charged off loans
        total_loans = len(self.loan_data)
        charged_off_percentage = (len(charged_off_loans) / total_loans) * 100

        # Calculate the total amount paid towards charged off loans
        total_paid_towards_charged_off = charged_off_loans['total_payment'].sum()

        return charged_off_percentage, total_paid_towards_charged_off
    
class ExpectedLoss: #Task 3
    '''
    - Calculate the expected loss of the loans marked as Charged Off.
    - Calculate the loss in revenue these loans would have generated for the company if they had finished 
    - their term. Visualise the loss projected over the remaining term of these loans.
    '''
    def __init__(self, loan_data):
        self.loan_data = loan_data

    def calculate_expected_loss(self):
        # Filter loans that are Charged Off
        charged_off_loans = self.loan_data[self.loan_data['loan_status'] == 'Charged Off'].copy()

        # Calculate the remaining term
        charged_off_loans['term (months)'] = charged_off_loans['term (months)'].astype(int)  # Convert term to int
        charged_off_loans.loc[:, 'remaining_term'] = (
            charged_off_loans['term (months)'] - (charged_off_loans['total_payment'] / charged_off_loans['instalment'])
        ).clip(lower=0)  # Ensure no negative remaining term , round off values

        # Calculate the projected loss
        charged_off_loans.loc[:, 'expected_loss'] = charged_off_loans['remaining_term'] * charged_off_loans['instalment']

        # Total expected loss
        total_expected_loss = charged_off_loans['expected_loss'].sum()
        
        # Calculate total revenue loss these loans would have generated
        charged_off_loans['total_revenue'] = charged_off_loans['remaining_term'] * charged_off_loans['instalment']
        total_revenue_loss = charged_off_loans['total_revenue'].sum()

        print(f"Total Expected Loss from Charged Off Loans: ${total_expected_loss:.2f}")
        print(f"Total Revenue Loss from Charged Off Loans: ${total_revenue_loss:.2f}")

        return total_expected_loss, total_revenue_loss

    def visualize_expected_loss(self):
        total_expected_loss, total_revenue_loss = self.calculate_expected_loss()

        # Plot total expected loss
        plt.figure(figsize=(8, 6))
        plt.bar(['Expected Loss', 'Revenue Loss'], [total_expected_loss, total_revenue_loss], color=['red', 'orange'])
        plt.title("Projected Losses from Charged Off Loans")
        plt.ylabel("Loss Amount ($)")
        plt.xticks(rotation=0)
        plt.grid(axis='y')

        # Show the plot
        plt.tight_layout()  # Improve spacing
        plt.show()


class PossibleLoss: #Task 4
    '''
    - There are customers who are currently behind with their loan payments. This subset of customers 
    represent a risk to company revenue.
    - What percentage do users in this bracket currently represent?
    - Calculate the total amount of customers in this bracket and how much loss the company would incur if 
    their status was changed to Charged Off.
    - What is the projected loss of these loans if the customer were to finish the full loan term?
    - If customers that are late on payments converted to Charged Off, what percentage of total expected 
    revenue do these customers and the customers who have already defaulted on their loan represent?
    '''
    def __init__(self, loan_data):
        self.loan_data = loan_data

    def calculate_possible_loss(self):
        # Filter customers who are behind on their payments
        late_customers = self.loan_data[self.loan_data['loan_status'].str.contains('Late')].copy()

        # Calculate the total number of late customers
        total_late_customers = late_customers.shape[0]

        # Calculate the total amount of loans for late customers
        total_late_amount = late_customers['loan_amount'].sum()

        # Calculate expected loss if they were to be charged off
        late_customers['term (months)'] = late_customers['term (months)'].astype(int)
        late_customers['remaining_term'] = late_customers['term (months)'] - (late_customers['total_payment'] / late_customers['instalment'])
        late_customers['expected_loss'] = late_customers['remaining_term'] * late_customers['instalment']
        total_expected_loss_late = late_customers['expected_loss'].sum()

        # Calculate total expected revenue
        total_revenue = self.loan_data['loan_amount'].sum()

        # Calculate the percentage of late customers compared to total customers
        total_customers = self.loan_data.shape[0]
        percentage_late_customers = (total_late_customers / total_customers) * 100 if total_customers > 0 else 0

        # Calculate expected loss for charged off loans
        charged_off_loans = self.loan_data[self.loan_data['loan_status'] == 'Charged Off'].copy()
        charged_off_loans['term (months)'] = charged_off_loans['term (months)'].astype(int)
        charged_off_loans['remaining_term'] = charged_off_loans['term (months)'] - (charged_off_loans['total_payment'] / charged_off_loans['instalment'])
        charged_off_loans['expected_loss'] = charged_off_loans['remaining_term'] * charged_off_loans['instalment']
        total_charged_off_loss = charged_off_loans['expected_loss'].sum()

        # Calculate the percentage of total expected revenue that late customers and charged off loans represent
        combined_loss = total_expected_loss_late + total_charged_off_loss
        percentage_combined_loss = (combined_loss / total_revenue) * 100 if total_revenue > 0 else 0

        # Print the results
        print(f"Total Late Customers: {total_late_customers}")
        print(f"Total Amount for Late Customers: ${total_late_amount:.2f}")
        print(f"Projected Loss from Late Customers if Charged Off: ${total_expected_loss_late:.2f}")
        print(f"Percentage of Customers Late on Payments: {percentage_late_customers:.2f}%")
        print(f"Percentage of Total Expected Revenue from Late and Charged Off Loans: {percentage_combined_loss:.2f}%")

        return {
            "total_late_customers": total_late_customers,
            "total_late_amount": total_late_amount,
            "total_expected_loss_late": total_expected_loss_late,
            "percentage_late_customers": percentage_late_customers,
            "percentage_combined_loss": percentage_combined_loss
        }

class LossIndicators: #Task 5
    '''
    - In this task, you will be analysing the data to visualise the possible indicators that a customer 
    will not be able to pay the loan.
    - You will want to compare columns which might be indicators against customers who have already stopped 
    paying and customers who are currently behind on payments.
    - Here are some example columns that might indicate that a user might not pay the loan:
        - Does the grade of the loan have an effect on customers not paying?
        - Is the purpose for the loan likely to have an effect?
        - Does the home_ownership value contribute to the likelihood a customer won't pay?
    - To help identify which columns will be of interest, first create a subset of these users.
    - Make the analysis and determine the columns are contributing to loans not being paid off and visualise any interesting indicators.
    - Compare these indicators between loans already charged off and loans that could change to charged off to check if these same factors apply to loans that have the potential to change to "Charged Off".
    '''
    def __init__(self, loan_data):
        self.loan_data = loan_data

    def analyze_loss_indicators(self):
        charged_off_loans = self.loan_data[self.loan_data['loan_status'] == 'Charged Off'].copy()
        late_customers = self.loan_data[self.loan_data['loan_status'].str.contains('Late')].copy()

        # Combine both groups for analysis
        charged_off_loans.loc[:, 'status'] = 'Charged Off'
        late_customers['status'] = 'Late'
        combined_data = pd.concat([charged_off_loans, late_customers], axis=0)

        # Set up visualizations for each indicator
        indicators = ['grade', 'purpose', 'home_ownership']
        plt.figure(figsize=(15, 10))
        
        for idx, indicator in enumerate(indicators):
            plt.subplot(2, 2, idx + 1)
            sns.countplot(data=combined_data, x=indicator, hue='status', order=combined_data[indicator].value_counts().index)
            plt.title(f'Comparison of {indicator} by Loan Status')
            plt.xticks(rotation=45)
            plt.xlabel(indicator)
            plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()

        # Analyze the impact of each indicator on loan status
        for indicator in indicators:
            if indicator in combined_data.columns:  # Check if the indicator exists
                print(f'\nAnalysis of {indicator}:\n')
                print(combined_data.groupby([indicator, 'status']).size().unstack().fillna(0))

    def run_analysis(self):
        self.analyze_loss_indicators()

if __name__ == "__main__":

    df = pd.read_csv(os.path.join('Source_Files', 'loan_payments_data_pretransformed.csv'))

    #Instantiating the classes
    loan_analysis = PaymentStateQuery(df)
    loss_calc = LossCalculation(df)
    expected_loss_calculator = ExpectedLoss(df)
    possible_loss_calculator = PossibleLoss(df)
    loss_indicators = LossIndicators(df)

    #Task 1
    loan_analysis.calculate_total_amount_due()
    total_recovered, total_due, percentage_recovered = loan_analysis.calculate_percentage_recovered()
    total_payment_in_6_months = loan_analysis.calculate_payment_in_6_months()
    loan_analysis.plot_recovered_vs_due(total_recovered, total_due, percentage_recovered)

    #Task 2
    percentage, total_paid = loss_calc.calculate_loss_percentage()
    print(f"Percentage of Charged Off Loans: {percentage:.2f}%")
    print(f"Total Amount Paid Towards Charged Off Loans: ${total_paid:.2f}")

    #Task 3
    expected_loss_calculator.visualize_expected_loss()

    #Task 4
    possible_loss_calculator.calculate_possible_loss()

    #Task 5
    loss_indicators.run_analysis()