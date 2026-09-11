import os
import io
import sys
import traceback
from typing import TypedDict, List, Optional

import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field


# ============================================================
# 1. LLM INITIALIZATION
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)


# ============================================================
# 2. STATE DEFINITION
# ============================================================

class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# 3. TOOLS
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return standard output or an error trace."""
    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code.replace("```python", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()
    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate 3 to 5 specific test scenarios for a coding task."""
    prompt = (
        "You are a Senior QA Engineer. Generate 3 to 5 highly specific "
        "test scenarios for the following coding task:\n\n"
        f"{task_description}\n\n"
        "Include standard cases and edge cases. Return them as a numbered list."
    )

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


# ============================================================
# 4. LANGGRAPH NODES
# ============================================================

def task_input_node(state: CrewState):
    """
    API-friendly replacement for the notebook's input() call.

    LangServe supplies the user's message in state["messages"].
    """
    messages = state.get("messages", [])

    if not messages:
        raise ValueError(
            "No task was provided. Send a user message in the request."
        )

    return {"next_step": "developer"}


def real_time_developer(state: CrewState):
    """Generate Python code for the requested coding task."""
    task = state["messages"][-1].content

    dev_prompt = (
        "Write a clean Python script to solve this coding task:\n"
        f"{task}\n\n"
        "Only return the code. Do not include explanation or markdown."
    )

    response = llm.invoke(dev_prompt)
    content = response.content

    # Gemini may return either a string or a list of content blocks.
    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(str(item))

        code_str = "\n".join(text_parts).strip()
    else:
        code_str = str(content).strip()

    return {"code": code_str}


def real_time_tester(state: CrewState):
    """Generate test scenarios and execute the generated Python code."""
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)
    cases_str = str(test_cases)

    execution_result = run_python_code.invoke(
        {"code": state.get("code", "")}
    )

    report = (
        "### GENERATED CODE:\n"
        f"{state.get('code', '')}\n\n"
        "### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS EVALUATED:\n"
        f"{cases_str}"
    )

    return {
        "report": report,
        "next_step": "done",
    }


def manager_decision_node(state: CrewState):
    """
    In the original notebook this node used input() for:
    'store / another'.

    That interactive behavior is not suitable for a REST API, so the
    API version completes the current request and returns the report.
    """
    return {"next_step": "done"}


def archiver_node(state: CrewState):
    """Mark the current workflow as archived/completed."""
    return {"next_step": "exit"}


# ============================================================
# 5. GRAPH CONSTRUCTION
# ============================================================

workflow = StateGraph(CrewState)

workflow.add_node("task_input", task_input_node)
workflow.add_node("developer", real_time_developer)
workflow.add_node("tester", real_time_tester)
workflow.add_node("manager_decision", manager_decision_node)
workflow.add_node("archiver", archiver_node)

workflow.add_edge(START, "task_input")

def route_from_input(state: CrewState):
    if state.get("next_step") == "exit":
        return END
    return "developer"

workflow.add_conditional_edges(
    "task_input",
    route_from_input,
)

workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager_decision")

def route_from_decision(state: CrewState):
    if state.get("next_step") == "archiver":
        return "archiver"
    return END

workflow.add_conditional_edges(
    "manager_decision",
    route_from_decision,
)

workflow.add_edge("archiver", END)

rt_app = workflow.compile()


# ============================================================
# 6. LANGSERVE INPUT/OUTPUT ADAPTER
# ============================================================

# Define the precise input schema so the Playground can render it
class AgentInput(BaseModel):
    input: str = Field(
        ..., 
        description="The coding task you want the agent to perform."
    )

def prepare_input(value: AgentInput):
    """
    Takes the validated AgentInput from LangServe and converts 
    it into the state format expected by LangGraph.
    """
    return {
        "messages": [
            HumanMessage(content=value.input)
        ]
    }

def prepare_output(state):
    """
    Return a simple JSON-friendly response while preserving the
    important LangGraph results.
    """
    if not isinstance(state, dict):
        return {"output": str(state)}

    return {
        "output": state.get("report", ""),
        "code": state.get("code", ""),
        "report": state.get("report", ""),
    }

# Explicitly attach the schema using .with_types()
agent_api = (
    RunnableLambda(prepare_input).with_types(input_type=AgentInput)
    | rt_app
    | RunnableLambda(prepare_output)
)


# ============================================================
# 7. FASTAPI + LANGSERVE
# ============================================================

app = FastAPI(
    title="LangGraph QA Coding Agent",
    version="1.0.0",
    description=(
        "A LangGraph workflow that generates Python code, "
        "creates QA test scenarios, executes the code, and "
        "returns a test report."
    ),
)

# IMPORTANT:
# The requested LangServe route path is exactly /Myaiagent
add_routes(
    app,
    agent_api,
    path="/Myaiagent",
)


@app.get("/")
def root():
    return {
        "message": "LangGraph + LangServe API is running.",
        "route": "/Myaiagent",
    }


# ============================================================
# 8. START SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
