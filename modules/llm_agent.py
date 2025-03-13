import sys
import pandas as pd
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

### CSV Dependencies ###
@dataclass
class CSVDependencies:
    csv_data: str

### CSV Analysis Result (No Insight Field) ###
class CSVAnalysisResult(BaseModel):
    question: str = Field(description="The user's query")
    answer: str = Field(description="Answer based on the CSV data")
    confidence: float = Field(description="Confidence score (0.0 - 1.0)", ge=0, le=1)

### Debugging Utility ###
def debug_log(message):
    """
    Ensures logs are flushed immediately for visibility during async execution.
    """
    sys.stdout.write(message + "\n")
    sys.stdout.flush()

### LLM Model and Agent Setup ###
model = OpenAIModel(
    model_name='llama3.1:8b',  # Ensure correct Llama 3 model
    provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

csv_agent = Agent(
    model=model,
    deps_type=CSVDependencies,
    result_type=CSVAnalysisResult,  # Structured Output
    system_prompt="You are a CSV analyst. Analyze CSV data to answer questions accurately."
)

### System Prompt Augmentation ###
@csv_agent.system_prompt
async def augment_csv_context(ctx: RunContext[CSVDependencies]) -> str:
    """
    Dynamically adds CSV metadata to the system prompt for better context.
    """
    csv_lines = len(ctx.deps.csv_data.splitlines())
    return f"The CSV includes {csv_lines} rows. Here are the first few rows for reference:\n{ctx.deps.csv_data}"

### CSV Summary Tool ###
@csv_agent.tool
async def csv_summary(ctx: RunContext[CSVDependencies]) -> str:
    """
    Provides a CSV preview to the LLM to enhance its ability to analyze the dataset.
    """
    return f"Dataset Preview:\n{ctx.deps.csv_data[:500]}"

### LLM Query Handler (Handles CSV queries) ###
async def ask_llm(question: str, file):
    """
    Handles CSV queries through the LLM and returns a structured output.
    """
    try:
        # Ensure CSV is loaded correctly
        df = pd.read_csv(file.name) if not isinstance(file, pd.DataFrame) else file

        if df.empty:
            return "Error: CSV is empty or malformed."

        # Use a small CSV preview for performance
        csv_data = df.head(5).to_string()

        # Prepare dependencies
        deps = CSVDependencies(csv_data=csv_data)

        debug_log(f"🚀 Sending Query: {question}")
        result = await csv_agent.run(question, deps=deps)

        # 🟢 Log the raw output for debugging
        debug_log(f"🔍 Raw LLM Output: {result}")

        if result and result.data:
            return result.data

        return "No valid output from the LLM."

    except Exception as e:
        return f"Error: {e}"
