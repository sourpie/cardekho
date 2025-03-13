from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from llm_service import generate_response

# Define our dependencies dataclass
@dataclass
class CSVQueryDependencies:
    """Dependencies required for CSV query processing"""
    dataframe: pd.DataFrame
    df_info: Dict[str, Any]
    question: str

# Define structured output model
class GraphParams(BaseModel):
    """Parameters for graph generation"""
    graph_type: str = Field(description="Type of graph to create: 'bar', 'line', 'scatter', 'histogram', etc.")
    x_axis: str = Field(description="Column to use for x-axis")
    y_axis: Union[str, List[str]] = Field(description="Column(s) to use for y-axis")
    title: Optional[str] = Field(None, description="Title for the graph")
    filters: Optional[Dict[str, Any]] = Field(None, description="Filters to apply before plotting")
    groupby: Optional[List[str]] = Field(None, description="Columns to group by before plotting")

class QueryResult(BaseModel):
    """Result of a CSV query"""
    answer: str = Field(description="The answer to the user's question")
    has_graph: bool = Field(description="Whether a graph should be generated")
    graph_params: Optional[GraphParams] = Field(None, description="Parameters for graph generation if needed")
    error: Optional[str] = Field(None, description="Error message if query processing failed")

# Create the Pydantic AI agent
csv_agent = Agent(
    "ollama:llama3.1:8b",  # Model name
    deps_type=CSVQueryDependencies,
    result_type=QueryResult,
    system_prompt=(
        "You are a data analysis assistant. Analyze the CSV data and answer questions about it. "
        "Provide clear, concise answers and determine if visualization would be helpful."
    ),
)

# Custom adapter for Ollama integration
def _ollama_adapter(model_name):
    """Create a custom adapter for Ollama integration with Pydantic AI"""
    from pydantic_ai.adapters.base import BaseAdapter
    from pydantic_ai.schema import LLMMessage, LLMChatCompletionResponse
        
    class OllamaAdapter(BaseAdapter):
        def __init__(self, model_name):
            self.model_name = model_name
        
        async def chat_completion(self, messages: List[LLMMessage]) -> LLMChatCompletionResponse:
            # Convert Pydantic AI messages to Ollama format
            system_prompt = None
            prompt = ""
            
            for msg in messages:
                if msg.role == "system":
                    system_prompt = msg.content
                elif msg.role == "user":
                    prompt += f"USER: {msg.content}\n"
                elif msg.role == "assistant":
                    prompt += f"ASSISTANT: {msg.content}\n"
            
            # Add final user prompt indicator
            prompt += "ASSISTANT: "
            
            # Get response from Ollama
            response = generate_response(prompt, system_prompt)
            
            # Create Pydantic AI response
            return LLMChatCompletionResponse(
                content=response,
                role="assistant"
            )
    
    # Extract the model name from the format "ollama:modelname"
    actual_model = model_name.split(":", 1)[1] if ":" in model_name else model_name
    return OllamaAdapter(actual_model)

# Register the custom adapter
csv_agent.adapter = _ollama_adapter(csv_agent.model)

# Add system prompt enhancement
@csv_agent.system_prompt
async def enhance_system_prompt(ctx: RunContext[CSVQueryDependencies]) -> str:
    """Add dataframe information to the system prompt"""
    df_info = ctx.deps.df_info
    return (
        f"The CSV data has {len(ctx.deps.dataframe)} rows and {len(ctx.deps.dataframe.columns)} columns.\n"
        f"Columns: {', '.join(df_info['columns'])}\n"
        f"Data types: {df_info['dtypes']}\n"
        f"Here are a few sample rows: {df_info['sample_rows']}"
    )

# Register data analysis tools
@csv_agent.tool
async def get_column_statistics(ctx: RunContext[CSVQueryDependencies], column_name: str) -> Dict[str, Any]:
    """Get statistics for a specific column"""
    df = ctx.deps.dataframe
    if column_name not in df.columns:
        return {"error": f"Column '{column_name}' not found"}
    
    series = df[column_name]
    try:
        if pd.api.types.is_numeric_dtype(series):
            return {
                "mean": float(series.mean()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
                "std": float(series.std()),
                "count": int(series.count()),
                "null_count": int(series.isna().sum())
            }
        else:
            return {
                "unique_values": series.nunique(),
                "most_common": series.value_counts().head(5).to_dict(),
                "count": int(series.count()),
                "null_count": int(series.isna().sum())
            }
    except Exception as e:
        return {"error": str(e)}

@csv_agent.tool
async def filter_data(
    ctx: RunContext[CSVQueryDependencies], 
    filters: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Filter the dataframe based on conditions
    
    Args:
        filters: Dict of {column: value} pairs to filter by
        
    Returns:
        Dict with row count and sample of filtered data
    """
    df = ctx.deps.dataframe
    filtered_df = df.copy()
    
    for col, value in filters.items():
        if col not in df.columns:
            return {"error": f"Column '{col}' not found"}
        
        filtered_df = filtered_df[filtered_df[col] == value]
    
    return {
        "row_count": len(filtered_df),
        "sample": filtered_df.head(3).to_dict(orient="records")
    }

@csv_agent.tool
async def determine_graph_type(
    ctx: RunContext[CSVQueryDependencies],
    columns: List[str]
) -> Dict[str, Any]:
    """
    Determine appropriate graph types for given columns
    
    Args:
        columns: List of column names to analyze
        
    Returns:
        Dict with recommended graph types
    """
    df = ctx.deps.dataframe
    invalid_columns = [col for col in columns if col not in df.columns]
    if invalid_columns:
        return {"error": f"Columns not found: {', '.join(invalid_columns)}"}
    
    recommendations = {}
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            recommendations[col] = ["histogram", "box", "line", "scatter"]
        elif pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 15:
            recommendations[col] = ["bar", "pie"]
        else:
            recommendations[col] = ["table"]
    
    return recommendations

# Main function to process queries
async def process_query(question: str, dataframe: pd.DataFrame) -> Dict[str, Any]:
    """
    Process a user query about the CSV data.
    
    Args:
        question: The user's question
        dataframe: The DataFrame containing the CSV data
        
    Returns:
        Dict: The result of processing the query with keys:
            - answer: Answer text
            - has_graph: Boolean indicating if graph should be generated
            - graph_params: Parameters for graph generation if needed
            - error: Error message if any
    """
    try:
        # Prepare dataframe information
        df_info = {
            "columns": list(dataframe.columns),
            "dtypes": {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
            "sample_rows": dataframe.head(3).to_dict(orient="records")
        }
        
        # Create dependencies
        deps = CSVQueryDependencies(
            dataframe=dataframe,
            df_info=df_info,
            question=question
        )
        
        # Run the agent
        result = await csv_agent.run(question, deps=deps)
        return result.data.dict()
    except Exception as e:
        return {
            "answer": f"An error occurred while processing your query: {str(e)}",
            "has_graph": False,
            "graph_params": None,
            "error": str(e)
        }