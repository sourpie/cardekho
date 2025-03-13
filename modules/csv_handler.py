import pandas as pd

class CSVHandler:
    def handle_csv(self, file):
        """
        Handles CSV file upload, validates the format, and parses the data.
        Returns a tuple containing:
        - The cleaned pandas DataFrame
        - A Markdown table representation of the top 5 rows
        """
        try:
            # Read the file content using the file path
            df = pd.read_csv(file.name)

            # Check if the DataFrame is empty
            if df.empty:
                raise ValueError("The CSV file is empty.")

            # Remove columns with no data or close to no data
            df = self._remove_empty_columns(df)

            # Fill missing data in columns with a few missing values
            df = self._fill_missing_data(df)

            # Get the top 5 rows as a Markdown table
            top_5_rows = self._df_to_markdown_table(df.head())

            return df, top_5_rows
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")
        except pd.errors.ParserError:
            raise ValueError("The CSV file is malformed or not a valid CSV.")
        except Exception as e:
            raise ValueError(f"Error handling CSV file: {e}")

    def _remove_empty_columns(self, df):
        """
        Removes columns with no data or close to no data.
        """
        # Drop columns with more than 90% missing values
        threshold = len(df) * 0.9
        df = df.dropna(axis=1, thresh=threshold)
        return df

    def _fill_missing_data(self, df):
        """
        Fills missing data in columns with a few missing values.
        """
        for column in df.columns:
            if df[column].isnull().sum() > 0:  # Check if the column has missing values
                if df[column].dtype == 'object':  # Categorical column
                    df[column] = df[column].fillna(df[column].mode()[0])  # Fill with mode
                else:  # Numerical column
                    df[column] = df[column].fillna(df[column].mean())  # Fill with mean
        return df

    def _df_to_markdown_table(self, df):
        """
        Converts a DataFrame to a Markdown table.
        """
        # Create the header row
        header = "| " + " | ".join(df.columns) + " |"
        # Create the separator row
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        # Create the data rows
        rows = []
        for _, row in df.iterrows():
            rows.append("| " + " | ".join(str(value) for value in row) + " |")
        # Combine into a Markdown table
        markdown_table = "\n".join([header, separator] + rows)
        return markdown_table