import os
import datetime
import pandas as pd

class GenericLogger:
    def __init__(self, file_name):
        """
        Initialize the logger by creating a unique directory based on the base_path,
        current date, and an incremental counter if needed.
        """
        extension_supported = [".csv", ".txt"]
        base_name, ext = os.path.splitext(file_name)
        if ext.lower() not in extension_supported:
            raise ValueError(f"File extension not supported. Supported extensions: {extension_supported}")
        
        self.base_name = base_name
        self.extension = ext

        self.log_file = self._update_filepath()
        with open(self.log_file, 'w') as f:
            f.write("")
    
    def _update_filepath(self):
        """
        Remove the extension from the base file name, append the current date and an optional counter,
        then add the extension back to form a unique file name.
        """
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        unique_base = f"{self.base_name}_{today_str}"
        unique_file = unique_base + self.extension
        counter = 1
        while os.path.exists(unique_file):
            unique_file = f"{unique_base}_{counter}{self.extension}"
            counter += 1
        return unique_file

class CSVLogger(GenericLogger):
    def __init__(self, file_name="log.csv"):
        """
        Initialize the CSVLogger by setting up a unique log directory and the CSV file path.
        """
        super().__init__(file_name)
        self.df = pd.DataFrame()
    
    def create_column(self, column_name, default_value=None):
        """
        Add a column to the DataFrame if it doesn't already exist.
        """
        if column_name not in self.df.columns:
            self.df[column_name] = default_value
    
    def log(self, data: dict):
        """
        Log a new row of data to the DataFrame.
        If any key in the provided data does not exist in the DataFrame, the column is added.
        """
        # Ensure all keys in data exist as columns
        for key in data:
            if key not in self.df.columns:
                self.create_column(key)
        # Append the new row
        row_df = pd.DataFrame([data])
        self.df = pd.concat([self.df, row_df], ignore_index=True)
    
    def save(self):
        """
        Save the current DataFrame to the CSV file.
        """
        self.df.to_csv(self.log_file, index=False)

class TextLogger(GenericLogger):
    def __init__(self, file_name="log.txt"):
        """
        Initialize the TextLogger by setting up a unique log directory and the text file path.
        """
        super().__init__(file_name)

    
    def log(self, text):
        """
        Log a line of text to the text file.
        """
        with open(self.log_file, "a") as f:
            f.write(f"{text}\n")

