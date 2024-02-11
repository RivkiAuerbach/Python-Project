
from collections import defaultdict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class DataSales:
    def __init__(self, data):
        self.data = pd.DataFrame(data)

    def _eliminate_duplicates(self):
        self.data = self.data.drop_duplicates()

    def _calculate_total_sales(self):
        self.data['Total Sales'] = self.data['Price'] * self.data['Quantity']

    def _calculate_total_sales_per_month(self):
        try:
            self.data['Date'] = pd.to_datetime(self.data['Date'])  # No need to specify format
        except ValueError:
            print(
                "Error: Date column is not in the correct format. Please ensure dates are in the format 'YYYY-MM-DD'.")
            return None

        self.data['Month'] = self.data['Date'].dt.to_period('M')
        monthly_sales = self.data.groupby('Month')['Total Sales'].sum()
        return monthly_sales.to_dict()

    def _identify_best_selling_product(self):
        best_selling_product = self.data.groupby('Product')['Total Sales'].sum().idxmax()
        return best_selling_product

    def _identify_month_with_highest_sales(self):
        monthly_sales = self._calculate_total_sales_per_month()
        month_with_highest_sales = max(monthly_sales, key=monthly_sales.get)
        return month_with_highest_sales

    def analyze_sales_data(self):
        self._eliminate_duplicates()
        self._calculate_total_sales()
        best_selling_product = self._identify_best_selling_product()
        month_with_highest_sales = self._identify_month_with_highest_sales()
        minimest_selling_product = self.data.groupby('Product')['Total Sales'].sum().idxmin()
        average_sales = self.data['Total Sales'].mean()
        analysis_result = {
            'best_selling_product': best_selling_product,
            'month_with_highest_sales': month_with_highest_sales,

        }
        return analysis_result

    def _add_additional_analysis(self, analysis_result):
        minimest_selling_product = self.data.groupby('Product')['Total Sales'].sum().idxmin()
        average_sales = self.data['Total Sales'].mean()
        analysis_result['minimest_selling_product'] = minimest_selling_product
        analysis_result['average_sales'] = average_sales
        return analysis_result



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

def add_90_percent_values_column(self):
    if 'Quantity' not in self.data.columns:
        print("Error: 'Quantity' column is required.")
        return

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
    return filtered_data


def divide_by_2(self):
    """
    Divide all values in the SalesData DataFrame by 2 for "BLACK FRIDAY".
    """
    if 'Price' in self.data.columns:
        self.data['BlackFridayPrice'] = self.data['Price'] / 2
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

    return stats
# --------------------------  👍✌️TASK THREE METHODS END ----------------------------------------------
#



















# import pandas as pd
# import numpy as np
#
# class DataSales:
#     def __init__(self, data):
#         self.data = data
#
#     def eliminate_duplicates(self):
#         """
#         Eliminate duplicate rows in the dataset to ensure data integrity and consistency.
#         Take care of null values.
#         """
#         # Remove duplicate rows
#         self.data = self.data.drop_duplicates()
#
#         # Handle null values
#         self.data = self.data.dropna()
#
#         return self.data
#
#     def calculate_total_sales(self):
#         """
#         Calculate the total quantity sold for each product.
#         """
#         # Group the data by product and sum the quantity sold
#         total_sales = self.data.groupby('Product')['Quantity'].sum()
#
#         return total_sales
#
#     def _calculate_total_sales_per_month(self):
#         """
#         Calculate the total quantity sold for each month.
#         """
#         # Convert the date column to datetime format
#         self.data['Date'] = pd.to_datetime(self.data['Date'])
#
#         # Extract the month from the date and group by month, summing the quantity sold
#         total_sales_per_month = self.data.groupby(self.data['Date'].dt.month)['Quantity'].sum()
#
#         return total_sales_per_month
#
#     def _identify_best_selling_product(self):
#         """
#         Identify the best-selling product based on total quantity sold.
#         """
#         # Group the data by product and sum the quantities, then find the product with the highest total quantity sold
#         best_selling_product = self.data.groupby('Product')['Quantity'].sum().idxmax()
#
#         return best_selling_product
#
#     def _identify_month_with_highest_sales(self):
#         """
#         Identify the month with the highest total sales.
#         """
#         # Calculate total sales per month
#         total_sales_per_month = self._calculate_total_sales_per_month()
#
#         # Find the month with the highest total sales
#         month_with_highest_sales = total_sales_per_month.idxmax()
#
#         return month_with_highest_sales
#
#     def analyze_sales_data(self):
#         """
#         Perform analysis using previously defined private methods and return a dictionary
#         with the analysis results.
#         """
#         # Call private methods to perform analysis
#         best_selling_product = self._identify_best_selling_product()
#         month_with_highest_sales = self._identify_month_with_highest_sales()
#
#         # Additional analysis
#         minimest_selling_product = self.data.groupby('Product')['Quantity'].sum().idxmin()
#         average_sales = self.data['Quantity'].mean()
#
#         # Construct the analysis dictionary
#         analysis_results = {
#             'best_selling_product': best_selling_product,
#             'month_with_highest_sales': month_with_highest_sales,
#             'minimest_selling_product': minimest_selling_product,
#             'average_sales': average_sales
#         }
#
#         return analysis_results
#
#     # def analyze_sales_data(self):
#     #     """
#     #     Perform analysis using previously defined private methods and return a dictionary
#     #     with the analysis results.
#     #     """
#     #     # Call private methods to perform analysis
#     #
#     #     best_selling_product = self._identify_best_selling_product()
#     #     print("hello")
#     #     month_with_highest_sales = self._identify_month_with_highest_sales()
#     #
#     #     # Additional analysis
#     #     minimest_selling_product = self.data.groupby('Product')['Sales'].sum().idxmin()
#     #     average_sales = self.data['Sales'].mean()
#     #
#     #     # Construct the analysis dictionary
#     #     analysis_results = {
#     #         'best_selling_product': best_selling_product,
#     #         'month_with_highest_sales': month_with_highest_sales,
#     #         'minimest_selling_product': minimest_selling_product,
#     #         'average_sales': average_sales
#     #     }
#     #     return analysis_results
#     def calculate_cumulative_sales(self):
#         if self.data is not None:  # Check if data is available
#             # Group the data by product and calculate the cumulative sum of sales
#             self.data['Cumulative_Quantity'] = self.data.groupby('Product')['Quantity'].cumsum()
#         else:
#             print("No data available for cumulative sales calculation.")
#     def add_90_percent_values_column(self):
#         if self.data is not None:  # Check if data is available
#             # Calculate the 90th percentile values of the 'Quantity' column
#             percentile_90 = np.percentile(self.data['Quantity'], 90)
#             # Create a new column for the 90th percentile values
#             self.data['90th_Percentile_Quantity'] = percentile_90
#             # Create a new column for the 'discount' and fill it with 90% of the values of the 'Quantity' column
#             self.data['Discount'] = 0.9 * self.data['Quantity']
#         else:
#             print("No data available for adding 90% values column.")
#
#     def change_index(self):
#         if self.data is not None:  # Check if data is available
#             # Set the index of the DataFrame to Customer ID multiplied by Price
#             self.data.index = self.data['Customer ID'] * self.data['Price']
#         else:
#             print("No data available for changing index.")
#
#     def split_and_concat(self):
#         if self.data is not None:  # Check if data is available
#             # Split the DataFrame into two
#             df1 = self.data.iloc[:len(self.data) // 2]
#             df2 = self.data.iloc[len(self.data) // 2:]
#             # Concatenate the two DataFrames along the columns axis
#             concatenated_df = pd.concat([df1, df2], axis=1)
#             return concatenated_df
#         else:
#             print("No data available for split and concatenate operation.")
#
#     def complex_data_transformation(self):
#         if self.data is not None:  # Check if data is available
#             # Transpose the DataFrame
#             transposed_df = self.data.T
#             return transposed_df
#         else:
#             print("No data available for complex data transformation.")
#
#     def group_with_function(self, column_do, column_use, func):
#         if self.data is not None:  # Check if data is available
#             # Group the DataFrame by specific columns and apply the function
#             grouped_data = self.data.groupby(column_do)[column_use].apply(func)
#             return grouped_data
#         else:
#             print("No data available for grouping with function.")
#
#     def locate(self, method, **kwargs):
#         if self.data is not None:  # Check if data is available
#             if method == 'row_by_index':
#                 # Locate specific row by index
#                 index = kwargs.get('index')
#                 row = self.data.iloc[index]
#                 return row
#             elif method == 'row_by_label':
#                 # Locate specific row by label
#                 label = kwargs.get('label')
#                 row = self.data.loc[label]
#                 return row
#             elif method == 'column_by_label':
#                 # Locate specific column by label
#                 label = kwargs.get('label')
#                 column = self.data[label]
#                 return column
#             elif method == 'column_by_index':
#                 # Locate specific column by index
#                 index = kwargs.get('index')
#                 column = self.data.iloc[:, index]
#                 return column







#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#
#class DataSales:
#    def __init__(self, data):
#        self.data = data
#
#    def eliminate_duplicates(self):
#        """
#        Eliminate duplicate rows in the dataset to ensure data integrity and consistency.
#        Take care of null values.
#        """
#        # Remove duplicate rows
#        self.data = self.data.drop_duplicates()
#
#        # Handle null values
#        self.data = self.data.dropna()
#
#        return self.data
#
#    def calculate_total_sales(self):
#        """
#        Calculate the total sales for each product.
#        """
#        # Group the data by product and sum the sales
#        total_sales = self.data.groupby('Product')['Sales'].sum()
#
#        return total_sales
#
#    def _calculate_total_sales_per_month(self):
#        """
#        Calculate the total sales for each month.
#        """
#        # Convert the date column to datetime format
#        self.data['Date'] = pd.to_datetime(self.data['Date'])
#
#        # Extract the month from the date and group by month, summing the sales
#        total_sales_per_month = self.data.groupby(self.data['Date'].dt.month)['Sales'].sum()
#
#        return total_sales_per_month
#
#    def _identify_best_selling_product(self):
#        """
#        Identify the best-selling product.
#        """
#        # Group the data by product and sum the sales, then find the product with the highest total sales
#        best_selling_product = self.data.groupby('Product')['Sales'].sum().idxmax()
#
#        return best_selling_product
#
#    def _identify_month_with_highest_sales(self):
#        """
#        Identify the month with the highest total sales.
#        """
#        # Calculate total sales per month
#        total_sales_per_month = self._calculate_total_sales_per_month()
#
#        # Find the month with the highest total sales
#        month_with_highest_sales = total_sales_per_month.idxmax()
#
#        return month_with_highest_sales
#
#    def analyze_sales_data(self):
#        """
#        Perform analysis using previously defined private methods and return a dictionary
#        with the analysis results.
#        """
#        # Call private methods to perform analysis
#        best_selling_product = self._identify_best_selling_product()
#        month_with_highest_sales = self._identify_month_with_highest_sales()
#
#        # Additional analysis
#        minimest_selling_product = self.data.groupby('Product')['Sales'].sum().idxmin()
#        average_sales = self.data['Sales'].mean()
#
#        # Construct the analysis dictionary
#        analysis_results = {
#            'best_selling_product': best_selling_product,
#            'month_with_highest_sales': month_with_highest_sales,
#            'minimest_selling_product': minimest_selling_product,
#            'average_sales': average_sales
#        }
#
#        return analysis_results
#
#    def calculate_cumulative_sales(self):
#        if self.data is not None:  # Check if data is available
#            # Group the data by product and calculate the cumulative sum of sales
#            self.data['Cumulative_Sales'] = self.data.groupby('Product')['Sales'].cumsum()
#        else:
#            print("No data available for cumulative sales calculation.")
#
#    def add_90_percent_values_column(self):
#        if self.data is not None:  # Check if data is available
#            # Calculate the 90th percentile values of the 'Quantity' column
#            percentile_90 = np.percentile(self.data['Quantity'], 90)
#            # Create a new column for the 90th percentile values
#            self.data['90th_Percentile_Quantity'] = percentile_90
#            # Create a new column for the 'discount' (הנחה) and fill it with 90% of the values of the 'Quantity' column
#            self.data['Discount'] = 0.9 * self.data['Quantity']
#        else:
#            print("No data available for adding 90% values column.")
#
#    # Method to change the indexing of the DataFrame to CustomerId*Price
#    def change_index(self):
#        if self.data is not None:  # Check if data is available
#            # Set the index of the DataFrame to CustomerId multiplied by Price
#            self.data.index = self.data['CustomerId'] * self.data['Price']
#        else:
#            print("No data available for changing index.")
#
#    # Method to split the DataFrame into two and concatenate them along a specific axis
#    def split_and_concat(self):
#        if self.data is not None:  # Check if data is available
#            # Split the DataFrame into two
#            df1 = self.data.iloc[:len(self.data)//2]
#            df2 = self.data.iloc[len(self.data)//2:]
#            # Concatenate the two DataFrames along the columns axis
#            concatenated_df = pd.concat([df1, df2], axis=1)
#            return concatenated_df
#        else:
#            print("No data available for split and concatenate operation.")
#
#    # Method to create a new DataFrame with transposed data
#    def complex_data_transformation(self):
#        if self.data is not None:  # Check if data is available
#            # Transpose the DataFrame
#            transposed_df = self.data.T
#            return transposed_df
#        else:
#            print("No data available for complex data transformation.")
#
#    # Method to group the DataFrame by specific columns and apply a function
#    def group_with_function(self, column_do, column_use, func):
#        if self.data is not None:  # Check if data is available
#            # Group the DataFrame by specific columns and apply the function
#            grouped_data = self.data.groupby(column_do)[column_use].apply(func)
#            return grouped_data
#        else:
#            print("No data available for grouping with function.")
#
#    # Method to locate specific rows, columns, or both using various methods
#    def locate(self, method, **kwargs):
#        if self.data is not None:  # Check if data is available
#            if method == 'row_by_index':
#                # Locate specific row by index
#                index = kwargs.get('index')
#                row = self.data.iloc[index]
#                return row
#            elif method == 'row_by_label':
#                # Locate specific row by label
#                label = kwargs.get('label')
#                row = self.data.loc[label]
#                return row
#            elif method == 'column_by_label':
#                # Locate specific column by label
#                label = kwargs.get('label')
#                column = self.data[label]
#                return column
#            elif method == 'column_by_index':
#                # Locate specific column by index
#                index = kwargs.get('index')
#                column = self.data.iloc[:, index]
#                return column
#            elif method == 'columns_and_rows_range':
#                # Locate specific columns and range of rows
#                columns = kwargs.get('columns')
#                start_row = kwargs.get('start_row')
#                end_row = kwargs.get('end_row')
#                data_subset = self.data.loc[start_row:end_row, columns]
#                return data_subset
#            elif method == 'rows_and_columns_range':
#                # Locate specific rows and range of columns
#                start_row = kwargs.get('start_row')
#                end_row = kwargs.get('end_row')
#                start_column = kwargs.get('start_column')
#                end_column =None
#


#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#---------------------LAST CLASSS-----------------------------------------------------------
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#from typing import List
#
#class DataSales:
#    def __init__(self, data):
#        self.data = data
#
#    def eliminate_duplicates(self):
#        """
#        Eliminate duplicate rows in the dataset to ensure data integrity and consistency.
#        Take care of null values.
#        """
#        # Remove duplicate rows
#        self.data = self.data.drop_duplicates()
#
#        # Handle null values
#        self.data = self.data.dropna()
#
#        return self.data
#
#    def calculate_total_sales(self):
#        """
#        Calculate the total sales for each product.
#        """
#        # Group the data by product and sum the sales
#        total_sales = self.data.groupby('Product')['Sales'].sum()
#
#        return total_sales
#
#    def _calculate_total_sales_per_month(self):
#        """
#        Calculate the total sales for each month.
#        """
#        # Convert the date column to datetime format
#        self.data['Date'] = pd.to_datetime(self.data['Date'])
#
#        # Extract the month from the date and group by month, summing the sales
#        total_sales_per_month = self.data.groupby(self.data['Date'].dt.month)['Sales'].sum()
#
#        return total_sales_per_month
#
#    def _identify_best_selling_product(self):
#        """
#        Identify the best-selling product.
#        """
#        # Group the data by product and sum the sales, then find the product with the highest total sales
#        best_selling_product = self.data.groupby('Product')['Sales'].sum().idxmax()
#
#        return best_selling_product
#
#    def _identify_month_with_highest_sales(self):
#        """
#        Identify the month with the highest total sales.
#        """
#        # Calculate total sales per month
#        total_sales_per_month = self._calculate_total_sales_per_month()
#
#        # Find the month with the highest total sales
#        month_with_highest_sales = total_sales_per_month.idxmax()
#
#        return month_with_highest_sales
#
#    def analyze_sales_data(self):
#        """
#        Perform analysis using previously defined private methods and return a dictionary
#        with the analysis results.
#        """
#        # Call private methods to perform analysis
#        best_selling_product = self._identify_best_selling_product()
#        month_with_highest_sales = self._identify_month_with_highest_sales()
#
#        # Additional analysis
#        minimest_selling_product = self.data.groupby('Product')['Sales'].sum().idxmin()
#        average_sales = self.data['Sales'].mean()
#
#        # Construct the analysis dictionary
#        analysis_results = {
#            'best_selling_product': best_selling_product,
#            'month_with_highest_sales': month_with_highest_sales,
#            'minimest_selling_product': minimest_selling_product,
#            'average_sales': average_sales
#        }
#
#        return analysis_results
#
#    def calculate_cumulative_sales(self):
#        """
#        Calculate the cumulative sum of sales for each product across months.
#        """
#        # Group the data by product and calculate the cumulative sum of sales
#        self.data['Cumulative_Sales'] = self.data.groupby('Product')['Sales'].cumsum()
#
#    def add_90_percent_values_column(self):
#        """
#        Create a new column in the DataFrame that contains the 90% values of the 'Quantity' column.
#        """
#        # Calculate the 90th percentile values of the 'Quantity' column
#        percentile_90 = np.percentile(self.data['Quantity'], 90)
#        # Create a new column for the 90th percentile values
#        self.data['90th_Percentile_Quantity'] = percentile_90
#        # Create a new column for the 'discount' (הנחה) and fill it with 90% of the values of the 'Quantity' column
#        self.data['Discount'] = 0.9 * self.data['Quantity']
#
#    def convert_date_format(self, date_columns: List = None):
#        """
#        Convert the 'Date' column in the DataSales DataFrame to datetime format.
#        """
#        if date_columns is None:
#            date_columns = ['Date']
#        for col in date_columns:
#            self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
#
#    def categorize_prices(self):
#        """
#        Create a new column in the DataFrame that categorizes price values into depends on prices.
#        """
#        # Define your categorization logic here
#        pass
#
#    def bar_chart_category_sum(self):
#        """
#        Plot a bar chart to represent the sum of quantities sold for each product.
#        """
#        # Group the data by product and calculate the sum of quantities sold for each product
#        category_sum = self.data.groupby('Product')['Quantity'].sum()
#        # Plotting a bar chart
#        category_sum.plot(kind='bar', figsize=(10, 6), ylabel='Quantity Sold', title='Sum of Quantities Sold for Each Product')
#        plt.xticks(rotation=45)
#        plt.show()
#
#    def calculate_mean_quantity(self):
#        """
#        Implement a NumPy array manipulation task to calculate the mean, median, and second max for Total column.
#        """
#        # Calculate the mean, median, and second max for the Total column
#        mean = np.mean(self.data['Total'])
#        median = np.median(self.data['Total'])
#        # Sorting the unique values of the Total column in descending order
#        sorted_unique = np.unique(self.data['Total'])[::-1]
#        # Finding the second maximum value
#        second_max = sorted_unique[1]
#        return mean, median, second_max
#
#    def filter_by_sellings_or_and(self):
#        """
#        Filter specific products by few things:
#        1. If number of selling more than 5 or number of selling == 0.
#        2. If the price above 300 $ and sold less than 2 times.
#        """
#        # Filter 1: Products sold more than 5 times or not sold at all
#        filter_1 = (self.data['Sales'] > 5) | (self.data['Sales'] == 0)
#        # Filter 2: Products with price above 300 $ and sold less than 2 times
#        filter_2 = (self.data['Price'] > 300) & (self.data['Sales'] < 2)
#        # Return the data that satisfies either filter 1 or filter 2
#        return self.data[filter_1 | filter_2]
#
#    # Additional methods for Task 4:
#
#    # Task 4: Data Handling and Cleaning
#    def convert_date_format(self, date_columns: List = None):
#        """
#        Convert the 'Date' column in the SalesData DataFrame to datetime format.
#        """
#        if date_columns is None:
#            date_columns = ['Date']
#        for column in date_columns:
#            self.data[column] = pd.to_datetime(self.data[column])
#
#    def categorize_prices(self):
#        """
#        Create a new column in the DataFrame that categorizes price values into depends on prices.
#        """
#        # Implementation of categorization based on price values
#        pass
#
#        # Task 20: Change Index
#
#    def change_index(self):
#        """
#        Change the indexing of the df, now it will be by CustomerId*Price.
#        """
#        self.data.set_index(['CustomerId', 'Price'], inplace=True)
#
#        # Task 21: Split and Concatenate
#
#    def split_and_concat(self):
#        """
#        Cut the df to 2 df's, and then concatenate them along a specific axis.
#        """
#        # Implementation of splitting and concatenating
#        pass
#
#        # Task 22: Complex Data Transformation
#
#    def complex_data_transformation(self):
#        """
#        Create a new DataFrame with transposed df.
#        """
#        # Implementation of complex data transformation
#        pass
#
#        # Task 23: Group with Function
#
#    def group_with_function(self, column_do, column_use, func):
#        """
#        Returns the groupby result from data frame, by specific demands and function.
#        """
#        # Implementation of grouping with function
#        pass
#
#        # Task 24: Locate
#
#    def locate_specific_row_by_index(self, index):
#        """
#        Locate a specific row from df by index.
#        """
#        # Implementation of locating a specific row by index
#        pass
#
#    def locate_specific_row_by_label(self, label):
#        """
#        Locate a specific row from df by label.
#        """
#        # Implementation of locating a specific row by label
#        pass
#
#    def locate_specific_column_by_label(self, label):
#        """
#        Locate a specific column from df by columns label.
#        """
#        # Implementation of locating a specific column by label
#        pass
#
#    def locate_specific_column_by_index(self, index):
#        """
#        Locate a specific column from df by index.
#        """
#        # Implementation of locating a specific column by index
#        pass
#
#    def locate_specific_columns_and_rows(self, columns, start_row, end_row):
#        """
#        Locate specific columns and range of rows.
#        """
#        # Implementation of locating specific columns and range of rows
#        pass
#
#    def locate_specific_rows_and_columns(self, rows, start_column, end_column):
#        """
#        Locate specific rows and range of columns.
#        """
#        # Implementation of locating specific rows and range of columns
#        pass
#
#        # Task 25: Filter by Mask
#
#    def filter_by_mask(self, mask_list, is_by_index=False):
#        """
#        Returns the df filters by the mask.
#        If is_by_index = True, use the mask for the indexes.
#        """
#        # Implementation of filtering by mask
#        pass
