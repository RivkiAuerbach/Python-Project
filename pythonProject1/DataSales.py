
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict


class DataSales:

    #--------------------------  👍TASK ONE METHODS START ----------------------------------------------
    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def _eliminate_duplicates(self):
        self.data = self.data.drop_duplicates()

    def _calculate_total_sales(self):
        self.data['Total Sales'] = self.data['Price'] * self.data['Quantity']
        plt.figure(figsize=(8, 6))
        plt.plot(self.data['Date'], self.data['Total Sales'], marker='o', linestyle='-')
        plt.title('Total Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.grid(True)
        plt.show()
    def _calculate_total_sales_per_month(self):
        try:
            self.data['Date'] = pd.to_datetime(self.data['Date'])  # No need to specify format
        except ValueError:
            print(
                "Error: Date column is not in the correct format. Please ensure dates are in the format 'YYYY-MM-DD'.")
            return None

        self.data['Month'] = self.data['Date'].dt.to_period('M')
        monthly_sales = self.data.groupby('Month')['Total Sales'].sum()
        plt.figure(figsize=(8, 6))
        monthly_sales.plot(kind='bar')
        plt.title('Total Sales by Month')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.show()
        return monthly_sales.to_dict()

    def _identify_best_selling_product(self):
        best_selling_product = self.data.groupby('Product')['Total Sales'].sum().idxmax()

        # Creating a box plot using Seaborn
        sns.boxplot(x='Product', y='Total Sales', data=self.data)
        plt.title('Total Sales Distribution by Product')
        plt.xlabel('Product')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.show()

        return best_selling_product

    # def _identify_month_with_highest_sales(self):
    #     monthly_sales = self._calculate_total_sales_per_month()
    #     month_with_highest_sales = max(monthly_sales, key=monthly_sales.get)
    #     return month_with_highest_sales

    def analyze_sales_data(self):
        self._eliminate_duplicates()
        self._calculate_total_sales()  # Ensure _calculate_total_sales is called first
        self._calculate_total_sales_per_month()  # Then call _calculate_total_sales_per_month
        best_selling_product = self._identify_best_selling_product()
        month_with_highest_sales = self._identify_month_with_highest_sales()
        minimest_selling_product = self.data.groupby('Product')['Total Sales'].sum().idxmin()
        average_sales = self.data['Total Sales'].mean()

        analysis_result = {
            'best_selling_product': best_selling_product,
            'month_with_highest_sales': month_with_highest_sales,
        }

        # Plotting the total sales for each product (Seaborn Bar Plot)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Product', y='Total Sales', data=self.data)
        plt.title('Total Sales by Product')
        plt.xlabel('Product')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.show()

        # Plotting a heatmap to visualize sales distribution across months (Seaborn Heatmap)
        plt.figure(figsize=(10, 6))
        sales_by_month = self.data.pivot_table(index='Product', columns='Month', values='Total Sales', aggfunc='sum')
        sns.heatmap(sales_by_month, cmap='Blues', annot=True, fmt='.1f', linewidths=.5)
        plt.title('Sales Distribution Across Months')
        plt.xlabel('Month')
        plt.ylabel('Product')
        plt.show()

        return analysis_result

    def _add_additional_analysis(self, analysis_result):
        minimest_selling_product = self.data.groupby('Product')['Total Sales'].sum().idxmin()
        average_sales = self.data['Total Sales'].mean()
        analysis_result['minimest_selling_product'] = minimest_selling_product
        analysis_result['average_sales'] = average_sales
        plt.figure(figsize=(8, 6))
        self.data.groupby('Product')['Total Sales'].sum().plot(kind='bar')
        plt.title('Total Sales by Product')
        plt.xlabel('Product')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.axhline(average_sales, color='r', linestyle='--', label='Average Sales')
        plt.legend()
        plt.show()
        return analysis_result

    def _identify_month_with_highest_sales(self):
        monthly_sales = self._calculate_total_sales_per_month()
        month_with_highest_sales = max(monthly_sales, key=monthly_sales.get)
        plt.figure(figsize=(8, 6))
        months_str= [str(month) for month in monthly_sales.keys()]
        plt.bar(months_str, monthly_sales.values(), color='skyblue')
        plt.title('Monthly Sales')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

        # Highlighting the month with the highest sales
        plt.axvline(x=str(month_with_highest_sales), color='red', linestyle='--', label='Month with Highest Sales')

        plt.legend()
        plt.grid(True)
        plt.show()
        return month_with_highest_sales




    # --------------------------  👍TASK ONE METHODS END ----------------------------------------------


    # --------------------------  👍✌️TASK THREE METHODS START : ----------------------------------------------

    def calculate_cumulative_sales(self):
        """
        Calculate the cumulative sum of sales for each product across months.
        """

        if 'Price' not in self.data.columns or 'Quantity' not in self.data.columns:
            print("Error: Columns 'Price' and 'Quantity' are required.")
            return

        self._calculate_total_sales()  # Utilize the method to calculate total sales
        self.data['Date'] = pd.to_datetime(self.data['Date'])  # Convert 'Date' column to datetime
        self.data['Month'] = self.data['Date'].dt.to_period('M')  # Extract month from date

        if 'Total Sales' not in self.data.columns:
            print("Error: 'Total Sales' column is required.")
            return

        cumulative_sales = self.data.groupby(['Product', 'Month'])['Total Sales'].sum().groupby('Product').cumsum()
        self.data['Cumulative Sales'] = cumulative_sales.values  # Add cumulative sales column to the DataFrame
        plt.figure(figsize=(8, 6))
        for product, group in self.data.groupby('Product'):
            # Convert 'Month' to string for plotting
            group['Month'] = group['Month'].astype(str)
            plt.plot(group['Month'], group['Cumulative Sales'], label=product)

        plt.title('Cumulative Sales Over Time')
        plt.xlabel('Month')
        plt.ylabel('Cumulative Sales')
        plt.legend()
        plt.grid(True)
        plt.show()

    def add_90_percent_values_column(self):
        """
        Add a new column to the dataset containing 90% of the values from the 'Quantity' column.
        """
        if 'Quantity' not in self.data.columns:
            print("Error: 'Quantity' column is required.")
            return

        # Calculate 90th percentile values for the 'Quantity' column
        percentile_90 = self.data['Quantity'].quantile(0.9)

        # Add a new column with 90th percentile values
        self.data['90th Percentile Quantity'] = percentile_90

        # Plot histogram to visualize the distribution of 'Quantity'
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.data, x='Quantity', bins=20, kde=True)
        plt.axvline(x=self.data['Quantity'].mean(), color='red', linestyle='--', label='Mean Quantity')
        plt.axvline(x=percentile_90, color='green', linestyle='--', label='90th Percentile Quantity')
        plt.title('Distribution of Quantity')
        plt.xlabel('Quantity')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        return self.data

        # Calculate 90th percentile of Quantity column for each product
        ninety_percent_values = self.data.groupby('Product')['Quantity'].quantile(0.9)

        # Map the calculated values back to the original DataFrame
        self.data['90% Quantity'] = self.data['Product'].map(ninety_percent_values)
    def bar_chart_category_sum(self):
         """
         Plot a bar chart to represent the sum of quantities sold for each product.
         """
         plt.figure(figsize=(10, 6))
         sns.barplot(x='Product', y='Quantity', data=self.data, estimator=sum)
         plt.title('Sum of Quantities Sold for Each Product')
         plt.xlabel('Product')
         plt.ylabel('Sum of Quantity')
         plt.xticks(rotation=45)
         plt.show()

    def calculate_mean_quantity(self):
        """
        Calculate mean, median, and second max for the 'Total Sales' column.
        """
        self._calculate_total_sales()  # Ensure 'Total Sales' column exists and is calculated

        total_sales_array = self.data['Total Sales'].to_numpy()
        mean_sales = np.mean(total_sales_array)
        median_sales = np.median(total_sales_array)
        sorted_sales = np.sort(total_sales_array)
        second_max_sales = sorted_sales[-2] if len(sorted_sales) >= 2 else None

        sns.histplot(data=self.data, x='Total Sales', bins=20, kde=True)
        plt.axvline(x=mean_sales, color='red', linestyle='--', label='Mean Sales')
        plt.axvline(x=median_sales, color='green', linestyle='--', label='Median Sales')
        if second_max_sales:
            plt.axvline(x=second_max_sales, color='orange', linestyle='--', label='Second Max Sales')
        plt.title('Distribution of Total Sales')
        plt.xlabel('Total Sales')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        return {
            'Mean Sales': mean_sales,
            'Median Sales': median_sales,
            'Second Max Sales': second_max_sales
        }
    def filter_by_sellings_or_and(self):
        """
        Filter specific products by number of selling more than 5 or number of selling equals to 0,
        and if the price is above 300 $ and sold less than 2 times.
        """
        filtered_data = self.data[(self.data['Quantity'] > 5) | (self.data['Quantity'] == 0) |
                                  ((self.data['Price'] > 300) & (self.data['Quantity'] < 2))]

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=filtered_data, x='Price', y='Quantity', hue='Product', size='Total Sales', sizes=(50, 200))
        plt.title('Filtered Products')
        plt.xlabel('Price')
        plt.ylabel('Quantity')
        plt.legend(title='Product', loc='upper right')
        plt.grid(True)
        plt.show()
        return filtered_data
    def  divide_by_2(self):
        """
        Divide all values in the SalesData DataFrame by 2 for "BLACK FRIDAY".
        """
        if 'Price' in self.data.columns:
            self.data['BlackFridayPrice'] = self.data['Price'] / 2

            # Plotting a histogram to visualize the distribution of prices after division by 2
            plt.figure(figsize=(10, 6))
            plt.hist(self.data['BlackFridayPrice'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            plt.title('Distribution of Prices after Division by 2')
            plt.xlabel('Price')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
        else:
            print("Error: 'Price' column is required.")

    def calculate_stats(self, columns=None):
        """
        Calculate the maximum, sum, absolute values, and cumulative maximum of the SalesData DataFrame for all
        columns, and for every column separately (depends on columns, if None: all)
        """
        if columns is None:
            columns = self.data.columns

        stats = {}
        for col in columns:
            if col in self.data.select_dtypes(include=[np.number]):
                col_data = self.data[col]
                stats[col] = {
                    'Maximum': col_data.max(),
                    'Sum': col_data.sum(),
                    'Absolute Values': col_data.abs().tolist(),
                    'Cumulative Maximum': col_data.cummax().tolist()
                }
        plt.figure(figsize=(8, 6))
        plt.plot(col_data.index, col_data.cummax() ,color='pink')
        plt.title(f'Cumulative Maximum for {col}')
        plt.xlabel('Index')
        plt.ylabel('Cumulative Maximum')
        plt.grid(True)
        plt.show()
        return stats
    # --------------------------  👍✌️TASK THREE METHODS END ----------------------------------------------



        # seborn-****
        # mat-*******

# def analyze_sales_data(self):-      Matplotlib
# calculate_cumulative_sales       - Seaborn













