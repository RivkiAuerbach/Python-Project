import csv
from DataSales import DataSales
#import openpyxl
from FileOperation import FileOperation
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import datetime
import os
import re
import sys

def main():
    # ------------------------- TASK 1 😝------------------------

    file_operation = FileOperation()

    # Reading data from a CSV file
    data = file_operation.read_csv("YafeNof.csv")

    if data is not None:
        print("Data read from CSV file:")
        print(data)
    else:
        print("Failed to read data from CSV file.")

    sample_data = {
        'Customer ID': [3, 5, 2, 6, 8, 10],
        'Date': ['2023-01-15', '2023-01-16', '2023-01-17', '2023-02-10', '2320-03-05', '2320-03-10'],
        'Product': ['Sidur', 'Teilim', 'Sidur', 'Chumash', 'Tanach', 'Gmara'],
        'Price': [60, 400, 50, 300, 80, 30],
        'Quantity': [3, 5, 10, 2, 20, 56]
    }

    file_operation.save_to_csv(pd.DataFrame(sample_data), "YafeNof.csv")

    # ------------------------- 😀 ------------------------

    # ------------------------- TASK 2 😉 ------------------------
    sample_data = {
        'Customer ID': [3, 5, 2, 6, 8, 10, 10],
        'Date': ['01-15-2023', '02-15-2023', '03-15-2023', '01-15-2023', '01-15-2023', '01-15-2023', '01-15-2023'],
        'Product': ['Sidur', 'Teilim', 'Sidur', 'Chumash', 'Tanach', 'Gmara', 'Gmara'],
        'Price': [60, 400, 50, 300, 80, 30, 30],
        'Quantity': [3, 5, 10, 2, 20, 56, 56]
    }

    analyzer = DataSales(sample_data)

    print("Data before eliminating duplicates:")
    print(analyzer.data)

    analyzer._eliminate_duplicates()

    print("\nData after eliminating duplicates:")
    print(analyzer.data)

    analyzer._calculate_total_sales()
    print("\nData after calculating total sales:")
    print(analyzer.data)

    # monthly_sales = analyzer._calculate_total_sales_per_month()
    # print("\nTotal sales per month:")
    # print(monthly_sales)

    monthly_sales = analyzer._calculate_total_sales_per_month()  # Accessing private method
    print("\nTotal sales per month:")
    print(monthly_sales)

    best_selling_product = analyzer._identify_best_selling_product()
    print("\nBest selling product:")
    print(best_selling_product)

    month_with_highest_sales = analyzer._identify_month_with_highest_sales()
    print("\nMonth with highest sales:")
    print(month_with_highest_sales)

    analysis_result = analyzer.analyze_sales_data()
    print("\nAnalysis result:")
    print(analysis_result)

    analysis_result2 = analyzer._add_additional_analysis(analysis_result)
    print(analysis_result2)


    # ------------------------- 😀 ------------------------

    # ------------------------- TASK 3 😘------------------------
    data = {
        'Product': ['A', 'B', 'A', 'C', 'B', 'A'],
        'Date': ['2022-01-01', '2022-01-02', '2022-02-01', '2022-02-05', '2022-03-01', '2022-03-05'],
        'Price': [10, 20, 15, 25, 30, 20],
        'Quantity': [5, 3, 7, 2, 4, 6]
    }
    sales_data = DataSales(data)
    sales_data.calculate_cumulative_sales()
    print(sales_data.data)

    # Call the method to add the 90% values column
    sales_data.add_90_percent_values_column()
    # Print the entire DataFrame without truncation
    print(sales_data.data)

    sales_data.bar_chart_category_sum()

    sales_data = DataSales(data)

    # Call the method to calculate mean, median, and second max for Total column
    mean_median_max = sales_data.calculate_mean_quantity()
    print("Mean, Median, and Second Max Sales:🤩")
    print(mean_median_max)

    # Call the method to filter specific products
    filtered_data = sales_data.filter_by_sellings_or_and()
    print("\nFiltered Data:👧")
    print(filtered_data)

    sales_data = DataSales(data)

    # Call the method to divide prices by 2 for BLACK FRIDAY
    sales_data.divide_by_2()

    # Print the DataFrame to verify the changes
    print("DataFrame after dividing prices by 2 for BLACK FRIDAY:👍")
    print(sales_data.data)

    # Call the method to calculate statistics for all columns
    all_stats = sales_data.calculate_stats()
    print("\nStatistics for all columns:")
    for column, stats in all_stats.items():
        print(f"{column}:")
        print(stats)

    # Call the method to calculate statistics for specific columns
    specific_columns_stats = sales_data.calculate_stats(columns=['Price', 'Quantity'])
    print("\nStatistics for 'Price' and 'Quantity' columns:😝")
    print(specific_columns_stats)

    # ----------------------------😁-----------------------------

     # ------------------------- TASK 4         !!! רשות 😡------------------------
    # # Convert date format and categorize prices
    # sales_data.convert_date_format()
    # sales_data.categorize_prices()
    # print("Data with Converted Date Format:")
    # print(sales_data.data)
    # print()
    #
    # sample_data = {
    #     'CustomerId': [1, 2, 3, 4, 5],
    #     'Price': [10, 20, 30, 40, 50],
    #     'Product': ['A', 'B', 'C', 'D', 'E'],
    #     'Quantity': [100, 200, 300, 400, 500],
    #     'Total': [1000, 4000, 9000, 16000, 25000]
    # }
    # df = pd.DataFrame(sample_data)
    # sales_data = DataSales(pd.DataFrame(sample_data))
    # analysis_results = sales_data.analyze_sales_data()
    # print("Analysis results:", analysis_results)
    # # Create an instance of DataSales
    # data_sales = DataSales(df)
    #
    # # Task 20: Change Index
    # data_sales.change_index()
    #
    # # Task 21: Split and Concatenate
    # data_sales.split_and_concat()
    #
    # # Task 22: Complex Data Transformation
    # data_sales.complex_data_transformation()
    #
    # # Task 23: Group with Function
    # data_sales.group_with_function(column_do='CustomerId', column_use='Total', func=np.sum)


    # Task 24: Locate
    # Implement calls to locate methods here

    # Task 25: Filter by Mask
    # Implement calls to filter_by_mask method here

#--------------------------😉-----------------------------------

# ------------------------ TASK 6 😍-----------------------------
  ##in the DataSales functions...

#-----------------------🌞-------------------------

# ------------------------- TASK 7 😍------------------------

# 1. Handling Errors
def handle_errors():
    try:
        # code that may raise errors
        result = 1 / 0
    except Exception as e:
        # Handle the error
        error_message = str(e)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"<rivki>, {current_time}> Error: {error_message} <rivki>")


# 2. Reading Additional Files
def read_additional_files():
    try:
        # Read CSV file
        additional_data_csv = pd.read_csv("YafeNof.csv")
        print("CSV File:")
        print(additional_data_csv)

        # Read Word file
        additional_data_word = pd.read_csv("YafeNof.csv")
        print("Word File:")
        print(additional_data_word)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")


# 3. Random Number Generator
def generate_random_number_and_max_payment(file_path, product_name):
    # קריאת קובץ CSV
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row)
    sales_count = 0
    max_payment = 0
    for row in data:
        if row['Product'] == product_name:
            sales_count += int(row['Quantity'])
            payment = int(row['Price']) * int(row['Quantity'])
            if payment > max_payment:
                max_payment = payment
    random_number = random.randint(sales_count, max_payment)
    return random_number, max_payment

# 4. Print Python Version
def print_python_version():
   print("Python Version:", sys.version)

# 5. Function with Arbitrary Number of Parameters
def process_parameters(*args, **kwargs):
    tagged_params = {}

    # Process positional arguments
    for arg in args:
        if isinstance(arg, int) or isinstance(arg, float):
            print(arg)
        elif isinstance(arg, str):
            print(f"Parameter without tag: {arg}")

    # Process keyword arguments
    for key, value in kwargs.items():
        if isinstance(value, int) or isinstance(value, float):
            tagged_params[key] = value
        elif isinstance(value, str):
            print(f"Parameter with tag ️'{key}': {value}")

    return tagged_params

# 6. Printing Table
def print_table():
    print("First 3 Rows:")
    print(data.head(3))
    print("Last 2 Rows:")
    print(data.tail(2))
    print("Random Row:")
    print(data.sample())


# 7. Looping through Table Elements
def loop_through_table():
    # Assuming 'data' is the main DataFrame
    for column in data.select_dtypes(include=['number']).columns:
        for value in data[column]:
            print(value)

#call the function:
# 1. Handling Errors
data=pd.read_csv("YafeNof.csv")
handle_errors()

# 2. Reading Additional Files
read_additional_files()

# 3. Random Number Generator
file_path = 'YafeNof.csv'
product_name = 'Sidur'
random_number, max_payment = generate_random_number_and_max_payment(file_path, product_name)
print(f"the random number {product_name}: {random_number}")
print(f"the max price {product_name}: {max_payment}")

# 4. Print Python Version
print_python_version()

 # 5. Function with Arbitrary Number of Parameters
process_parameters(1, "tag1", param2=2, param3="value3")

 # 6. Printing Table
print_table()

# 7. Looping through Table Elements
loop_through_table()

# ------------------------- 😀------------------------

# ------------------------- TASK 8 🤩------------------------
def task1_check_file_existence(file_path):
    if not os.path.exists(file_path):
        # Create the file if it does not exist
        with open(file_path, 'w') as file:
            pass
        print(f"File '{file_path}' created successfully.")
    return file_path

def task2_read_users_to_generator(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

def task3_read_users_to_array(file_path, exclude_first_10_percent=False):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if exclude_first_10_percent:
            lines = lines[int(len(lines) * 0.1):]
        usernames = [line.strip() for line in lines]
    return usernames

def task4_create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    return folder_path

def task5_validate_emails(emails):
    valid_emails = []
    invalid_emails = []

    for email in emails:
        if "@" in email and "." in email.split("@")[1]:
            valid_emails.append(email)
        else:
            invalid_emails.append(email)

    return valid_emails, invalid_emails

def task6_filter_gmail_addresses(email_addresses):
    gmail_addresses = [email for email in email_addresses if email.endswith('@gmail.com')]
    return gmail_addresses

def task7_convert_name_to_ascii(name):
    ascii_name = ', '.join(str(ord(char)) for char in name)
    return ascii_name


def task8_check_username(user_name, name_list):
    # Check if the username is in the list
    if user_name in name_list:
        print(f"The username {user_name} is in the list.")

        # Convert username to ASCII format
        ascii_name = ''.join(str(ord(char)) for char in user_name)

        # Convert ASCII format back to string
        converted_name = ''.join(chr(int(ascii_name[i:i + 2])) for i in range(0, len(ascii_name), 2))

        print(f"Converted name: {converted_name}")

        # Count occurrences of 'A' in the username
        count_A = user_name.count('A')
        print(f"The letter 'A' appears {count_A} times in the username.")
    else:
        print(f"The username {user_name} is not in the list.")


def task9_convert_to_uppercase(usernames):
    usernames_uppercase = [name.upper() for name in usernames]
    return usernames_uppercase


def task10_calculate_profit(customers_list):
    total_profit = 0
    group_size = 8
    additional_payment = 50

    # Iterate over the customers list
    for i, customer in enumerate(customers_list):
        # Check if the customer index is divisible by 8 without remainder
        if (i + 1) % group_size == 0:
            total_profit += customer
        else:
            # Check if the current customer index is beyond the last multiple of 8
            if (i + 1) > ((i // group_size) + 1) * group_size:
                total_profit += customer + additional_payment

    return total_profit


# Example usage:

# Task 1: Check file existence
file_path = "users.txt"
file_path = task1_check_file_existence(file_path)

# Task 2: Read usernames to generator
users_generator = task2_read_users_to_generator(file_path)
for user in users_generator:
    print(user)

# Task 3: Read usernames to array
usernames_array = task3_read_users_to_array(file_path)
print(usernames_array)

# Task 4: Create folder if not exists
folder_path = "data"
folder_path = task4_create_folder_if_not_exists(folder_path)

# Task 5: Read and validate email addresses
emails = ["user1@example.com", "user2@example", "user3@gmail.com"]
valid_emails, invalid_emails = task5_validate_emails(emails)
print("Valid emails:")
for email in valid_emails:
    print(email)
print("\nInvalid emails:")
for email in invalid_emails:
    print(email)

# Task 6: Filter Gmail addresses
gmail_addresses = task6_filter_gmail_addresses(valid_emails)
print("Gmail addresses:", gmail_addresses)

# Task 7: Convert name to ASCII
name = "John Doe"
ascii_name = task7_convert_name_to_ascii(name)
print("ASCII name:", ascii_name)

# Task 8: Check if a username exists in a list
usernames_list = ['Alice', 'Bob', 'Charlie', 'David']
username = input("Enter your username: ")
task8_check_username(username, usernames_list)

# Task 9: Convert all usernames to uppercase
uppercase_usernames = task9_convert_to_uppercase(usernames_list)
print("Uppercase usernames:", uppercase_usernames)

# Task 10: Calculate total payment for the team
customers_list = [200, 24, 88, 20, 76, 88, 4, 43, 19, 5]
total_profit = task10_calculate_profit(customers_list)
print(f"Total profit: {total_profit} shekels")

# ------------------------- 🤩------------------------

# ------------------------- TASK 9 🤩------------------------
#in PDF ...

# ------------------------- 😀------------------------

if __name__ == "__main__":
    main()








