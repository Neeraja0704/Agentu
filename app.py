import os
import uvicorn
from typing import TypedDict, List, Optional, Any

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import requests
import json


# ============================================================
# 1. TOOLS
# ============================================================

def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""
    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali"
    }
    return movies.get(genre.lower(), "No movies found for that genre")


def change__to_f(temp_c: float) -> float:
    """Converts Celsius temperature to Fahrenheit."""
    return temp_c * 1.8 + 32


def get_weather(city: str) -> str:
    """Get current temperature for a given city name."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_response = requests.get(
        geo_url, params={"name": city, "count": 1}, timeout=15
    ).json()

    if "results" not in geo_response:
        return f"Could not find weather data for city: {city}"

    location = geo_response["results"][0]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "current": "temperature_2m,weather_code",
        "temperature_unit": "celsius"
    }

    current = requests.get(
        weather_url, params=weather_params, timeout=15
    ).json()["current"]

    return json.dumps({
        "resolved_city": location["name"],
        "temperature_celsius": current["temperature_2m"],
        "weather_code": current["weather_code"]
    })


# ============================================================
# 2. GEMINI
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================

class CrewState(TypedDict):
    input: str
    messages: List[Any]
    developer_output: Optional[str]
    tester_output: Optional[str]
    manager_output: Optional[str]


# ============================================================
# 4. LANGGRAPH NODES
# ============================================================

def task_input_node(state: CrewState):
    return {
        "messages": [HumanMessage(content=state["input"])]
    }


def developer_node(state: CrewState):
    user_input = state["input"]

    prompt = f"""
You are the Developer state of a multi-state application.

The application is ONLY authorized to answer questions about:
1. Indian weather
2. Indian cinema/movies

User request:
{user_input}

For an authorized request, provide the best answer.
For anything outside Indian weather and cinema, say exactly:
I am not authorized to answer questions outside of Indian weather and cinema.
"""

    response = llm_flash.invoke(prompt)

    if isinstance(response.content, list):
        parts = []
        for item in response.content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        output = "\n".join(parts)
    else:
        output = str(response.content)

    return {"developer_output": output}


def tester_node(state: CrewState):
    developer_output = state.get("developer_output", "")

    if developer_output:
        result = "PASS: Developer response generated successfully."
    else:
        result = "FAIL: Developer response was empty."

    return {"tester_output": result}


def manager_node(state: CrewState):
    # Preserve the same user-facing answer while making Manager
    # a real LangGraph state/node.
    return {
        "manager_output": state.get("developer_output", "")
    }


# ============================================================
# 5. GRAPH CONSTRUCTION
# ============================================================

workflow = StateGraph(CrewState)

workflow.add_node("task_input", task_input_node)
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_node("manager", manager_node)

workflow.add_edge(START, "task_input")
workflow.add_edge("task_input", "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", "manager")
workflow.add_edge("manager", END)

graph = workflow.compile()


# ============================================================
# 6. LANGSERVE INPUT / OUTPUT
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


def format_for_graph(x):
    user_input = x["input"] if isinstance(x, dict) else x.input

    return {
        "input": user_input,
        "messages": [],
        "developer_output": None,
        "tester_output": None,
        "manager_output": None
    }


def format_graph_output(state):
    if not isinstance(state, dict):
        return str(state)

    return state.get("manager_output", "")


formatted_agent_chain = (
    RunnableLambda(format_for_graph)
    | graph
    | RunnableLambda(format_graph_output)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 7. FASTAPI + LANGSERVE
# ============================================================

app = FastAPI(
    title="Movie and Weather Agent",
    version="1.0",
    description="LangGraph Developer -> Tester -> Manager workflow served through LangServe."
)


@app.get("/")
def root():
    return {
        "message": "Server is running. Visit /agent/playground/ to chat, or /docs for the API."
    }


add_routes(app, formatted_agent_chain, path="/agent")


# ============================================================
# 8. RENDER
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
