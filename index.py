import gradio as gr
import pandas as pd
from modules.csv_handler import CSVHandler
from modules.llm_agent import ask_llm
from modules.graph_plotter import GraphPlotter

# Display CSV Preview
def display_top_5_rows(file):
    try:
        csv_handler = CSVHandler()
        df, top_5_rows = csv_handler.handle_csv(file)
        return top_5_rows
    except Exception as e:
        return f"Error: {str(e)}"

# Process CSV and Answer Questions

async def process_csv(file, question):
    """
    Processes CSV and queries the LLM for structured output.
    """
    try:
        if not question.strip():
            return "Error: Please enter a valid question."

        # Query the CSV Analyst
        result = await ask_llm(question, file)

        if isinstance(result, dict):
            return f"🗨️ **Answer:** {result['answer']}\n✅ **Confidence:** {result['confidence']:.2f}"
        else:
            return result

    except Exception as e:
        return f"Error processing query: {e}"

# Graph Generation
def generate_graph(file, question, graph_type, columns):
    try:
        if not columns:
            return "Error: Please select at least one column for plotting."

        csv_handler = CSVHandler()
        df, _ = csv_handler.handle_csv(file)

        graph_plotter = GraphPlotter()
        plot = graph_plotter.plot_data(df, question, graph_type, columns)

        return plot
    except Exception as e:
        return f"Error generating graph: {str(e)}"

# Update Columns in Graph UI
def update_columns(file):
    try:
        df = pd.read_csv(file.name)
        numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
        return gr.update(choices=numerical_columns)
    except Exception:
        return gr.update(choices=[])

# Gradio UI
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# CSV Analysis and Graph Generator")

        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload CSV")
                top_5_output = gr.Textbox(label="CSV Preview", interactive=False)

        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(label="Ask a Question")
                submit_button = gr.Button("Submit")
                answer_output = gr.Textbox(label="Answer")

            with gr.Column():
                graph_type_input = gr.Dropdown(choices=["line", "bar", "scatter"], label="Graph Type")
                columns_input = gr.CheckboxGroup(label="Select Columns")
                graph_button = gr.Button("Generate Graph")
                graph_output = gr.Plot(label="Graph")

        # Link functions to UI
        file_input.change(display_top_5_rows, inputs=file_input, outputs=top_5_output)
        file_input.change(update_columns, inputs=file_input, outputs=columns_input)

        submit_button.click(process_csv, inputs=[file_input, question_input], outputs=answer_output)
        graph_button.click(generate_graph, inputs=[file_input, question_input, graph_type_input, columns_input], outputs=graph_output)

    demo.launch()

if __name__ == "__main__":
    main()