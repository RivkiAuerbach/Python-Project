import pandas as pd

class FileOperation:
    def read_csv(self, file_path: str):
        """
        Read data from a CSV file located at the specified file path.

        Parameters:
        file_path (str): Path to the CSV file.

        Returns:
        DataFrame: Data read from the CSV file.
        """
        try:
            data = pd.read_csv(file_path)
            return data
        except FileNotFoundError as e:
            print(f"Error: File not found at path '{file_path}'.")
            return None
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None


#def save_to_csv(self, data, file_name: str):
#    """
#    Save the provided data to a new CSV file with the given file name.
#    If the file already exists, it will be overwritten.
#
#    Parameters:
#    data: Data to be saved to CSV.
#    file_name (str): Name of the CSV file to be saved.
#    """
#    try:
#        if isinstance(data, pd.DataFrame):
#            data.to_csv(file_name, index=False)
#            print(f"Data saved to {file_name} successfully.")
#            return True
#        else:
#            print("Invalid data type. Please provide a Pandas DataFrame.")
#            return False
#    except Exception as e:
#        print(f"Error saving data to CSV file: {e}")
#        return False

    def save_to_csv(self, data, file_name: str):
        """
        Save the provided data to a new CSV file with the given file name.
        If the file already exists, it will be overwritten.

        Parameters:
        data: Data to be saved to CSV.
        file_name (str): Name of the CSV file to be saved.
        """
        try:
            if isinstance(data, pd.DataFrame):
                data.to_csv(file_name, index=False)
                print(f"Data saved to {file_name} successfully.")
                return True
            else:
                print("Invalid data type. Please provide a Pandas DataFrame.")
                return False
        except Exception as e:
            print(f"Error saving data to CSV file: {e}")
            return False