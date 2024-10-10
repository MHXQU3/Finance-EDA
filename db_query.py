import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PaymentStateQuery:
    def __init__(self, cleaned_loan_data):
        self.cleaned_loan_data = cleaned_loan_data

    # Calculate total amount due (loan amount + interest)
    def calculate_total_amount_due(self):
        self.cleaned_loan_data['total_amount_due'] = self.cleaned_loan_data['loan_amount'] + (
            self.cleaned_loan_data['loan_amount'] * self.cleaned_loan_data['int_rate'] / 100)

    # Calculate percentage of loans recovered
    def calculate_percentage_recovered(self):
        total_recovered = self.cleaned_loan_data['recoveries'].sum()
        total_due = self.cleaned_loan_data['total_amount_due'].sum()
        percentage_recovered = (total_recovered / total_due) * 100
        print(f"Percentage of loans recovered: {percentage_recovered:.2f}%")
        return total_recovered, total_due, percentage_recovered

    # Calculate total amount to be paid back in 6 months
    def calculate_payment_in_6_months(self):
        self.cleaned_loan_data['monthly_payment'] = self.cleaned_loan_data['total_amount_due'] / self.cleaned_loan_data['term (months)']
        self.cleaned_loan_data['payment_in_6_months'] = self.cleaned_loan_data['monthly_payment'] * 6
        total_payment_in_6_months = self.cleaned_loan_data['payment_in_6_months'].sum()
        print(f"Total payment to be paid back in 6 months: {total_payment_in_6_months:.2f}")
        return total_payment_in_6_months

    # Plot recovered vs total due
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

class LossCalculation:
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
    
class ExpectedLoss:
    def __init__(self, cleaned_loan_data):
        self.cleaned_loan_data = cleaned_loan_data

    def calculate_expected_loss(self):
        # Filter loans that are Charged Off
        charged_off_loans = self.cleaned_loan_data[self.cleaned_loan_data['loan_status'] == 'Charged Off'].copy()

        # Calculate the remaining term using .loc to avoid the warning
        charged_off_loans.loc[:, 'remaining_term'] = charged_off_loans['term (months)'] - (charged_off_loans['total_payment'] / charged_off_loans['instalment'])

        # Calculate the projected loss: remaining term * monthly instalment using .loc to avoid the warning
        charged_off_loans.loc[:, 'expected_loss'] = charged_off_loans['remaining_term'] * charged_off_loans['instalment']

        # Total expected loss
        total_expected_loss = charged_off_loans['expected_loss'].sum()
        print(f"Total Expected Loss from Charged Off Loans: ${total_expected_loss:.2f}")

        return charged_off_loans, total_expected_loss

    def visualize_expected_loss(self):
        charged_off_loans, total_expected_loss = self.calculate_expected_loss()

        # Sort loans by expected loss in descending order
        charged_off_loans = charged_off_loans.sort_values(by='expected_loss', ascending=False)

        # Limit the number of loans displayed (e.g., top 20 largest losses)
        top_loans = charged_off_loans.head(20)

        # Plot the top 20 projected losses over the remaining term
        plt.figure(figsize=(12, 8))
        plt.bar(top_loans['id'], top_loans['expected_loss'], color='red')
        plt.title(f"Top 20 Projected Losses Over Remaining Term for Charged Off Loans\nTotal Expected Loss: ${total_expected_loss:.2f}")
        plt.xlabel("Loan ID")
        plt.ylabel("Expected Loss ($)")
        plt.xticks(rotation=90)
        plt.grid(axis='y')

        # Show the plot
        plt.tight_layout()  # Improve spacing
        plt.show()


class PossibleLoss:
    def __init__(self, cleaned_loan_data):
        self.cleaned_loan_data = cleaned_loan_data

    def calculate_possible_loss(self):
        # Filter customers who are behind on their payments
        late_customers = self.cleaned_loan_data[self.cleaned_loan_data['loan_status'] == 'Late'].copy()

        # Calculate the total number of late customers
        total_late_customers = late_customers.shape[0]

        # Calculate the total amount of loans for late customers
        total_late_amount = late_customers['loan_amount'].sum()

        # Calculate expected loss if they were to be charged off
        late_customers['remaining_term'] = late_customers['term (months)'] - (late_customers['total_payment'] / late_customers['instalment'])
        late_customers['expected_loss'] = late_customers['remaining_term'] * late_customers['instalment']
        total_expected_loss_late = late_customers['expected_loss'].sum()

        # Calculate total expected revenue
        total_revenue = self.cleaned_loan_data['loan_amount'].sum()

        # Calculate the percentage of late customers compared to total customers
        total_customers = self.cleaned_loan_data.shape[0]
        percentage_late_customers = (total_late_customers / total_customers) * 100

        # Calculate expected loss for charged off loans
        charged_off_loans = self.cleaned_loan_data[self.cleaned_loan_data['loan_status'] == 'Charged Off'].copy()
        charged_off_loans['remaining_term'] = charged_off_loans['term (months)'] - (charged_off_loans['total_payment'] / charged_off_loans['instalment'])
        charged_off_loans['expected_loss'] = charged_off_loans['remaining_term'] * charged_off_loans['instalment']
        total_charged_off_loss = charged_off_loans['expected_loss'].sum()

        # Calculate the percentage of total expected revenue that late customers and charged off loans represent
        combined_loss = total_expected_loss_late + total_charged_off_loss
        percentage_combined_loss = (combined_loss / total_revenue) * 100

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

class LossIndicators:
    def __init__(self, cleaned_loan_data):
        self.cleaned_loan_data = cleaned_loan_data

    def analyze_loss_indicators(self):
        # Create subsets for charged off loans and late customers
        charged_off_loans = self.cleaned_loan_data[self.cleaned_loan_data['loan_status'] == 'Charged Off']
        late_customers = self.cleaned_loan_data[self.cleaned_loan_data['loan_status'] == 'Late']

        # Combine both groups for analysis
        charged_off_loans['status'] = 'Charged Off'
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
            print(f'\nAnalysis of {indicator}:\n')
            print(combined_data.groupby([indicator, 'status']).size().unstack().fillna(0))

    def run_analysis(self):
        self.analyze_loss_indicators()

# Usage
df = pd.read_csv(os.path.join('Source_Files', 'loan_payments_data.csv'))  # Replace with actual cleaned data path

#Instantiating the classes
loan_analysis = PaymentStateQuery(df)
loss_calc = LossCalculation(df)
expected_loss_calculator = ExpectedLoss(df)
possible_loss_calculator = PossibleLoss(df)
loss_indicators = LossIndicators(df)

#loan analysis
loan_analysis.calculate_total_amount_due()
total_recovered, total_due, percentage_recovered = loan_analysis.calculate_percentage_recovered()
total_payment_in_6_months = loan_analysis.calculate_payment_in_6_months()
loan_analysis.plot_recovered_vs_due(total_recovered, total_due, percentage_recovered)

#loss calculation
percentage, total_paid = loss_calc.calculate_loss_percentage()
print(f"Percentage of Charged Off Loans: {percentage:.2f}%")
print(f"Total Amount Paid Towards Charged Off Loans: ${total_paid:.2f}")

#Expected loss
expected_loss_calculator.visualize_expected_loss()

#Possible Loss
possible_loss_calculator.calculate_possible_loss()

#Loss Indicators
loss_indicators.run_analysis()