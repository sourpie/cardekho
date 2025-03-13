import matplotlib.pyplot as plt
import pandas as pd

class GraphPlotter:
    def plot_data(self, df, question, graph_type="line", columns=None):
        """
        Generates a graph based on the CSV data and the user's question.
        """
        try:
            plt.figure()

            # Debugging: Print the DataFrame and graph type
            print(f"DataFrame:\n{df}")
            print(f"Graph Type: {graph_type}")

            # Select only numerical columns for plotting
            numerical_df = df.select_dtypes(include=['number'])

            # If no numerical columns are found, raise an error
            if numerical_df.empty:
                raise ValueError("The DataFrame has no numerical columns for plotting.")

            # If specific columns are provided, filter the DataFrame
            if columns:
                numerical_df = numerical_df[columns]

            # Generate the graph based on the user's choice
            if graph_type == "line":
                numerical_df.plot(kind='line')
            elif graph_type == "bar":
                numerical_df.plot(kind='bar')
            elif graph_type == "scatter":
                if len(numerical_df.columns) < 2:
                    raise ValueError("Scatter plot requires at least two numerical columns.")
                numerical_df.plot(kind='scatter', x=numerical_df.columns[0], y=numerical_df.columns[1])
            else:
                numerical_df.plot()  # Default to line plot

            plt.title(question)
            return plt
        except Exception as e:
            raise Exception(f"Error plotting graph: {e}")