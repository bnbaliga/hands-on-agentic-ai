
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
import os
import requests

load_dotenv()

# Open-Meteo returns conditions as WMO codes, so map the common ones to words.
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "drizzle",
    55: "dense drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail",
}

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Use this for any question about current weather, temperature, or
    conditions in a named place. Example city: "Tokyo"
    """
    try:
        # 1. Turn the city name into coordinates.
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=15,
        ).json()

        results = geo.get("results")
        if not results:
            return f"Error: could not find a place called '{city}'."

        place = results[0]

        # 2. Fetch current conditions for those coordinates.
        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            },
            timeout=15,
        ).json()

        now = weather.get("current", {})
        condition = WEATHER_CODES.get(now.get("weather_code"), "unknown conditions")

        return (
            f"{place['name']}, {place.get('country', '')}: {condition}, "
            f"{now.get('temperature_2m')}C, "
            f"humidity {now.get('relative_humidity_2m')}%, "
            f"wind {now.get('wind_speed_10m')} km/h"
        )

    except Exception as exc:
        return f"Error: weather lookup failed ({exc})."

def agentic():

    llm = ChatGoogleGenerativeAI(
        model='gemini-3.5-flash',
        google_api_key=os.getenv('GOOGLE_API_KEY'),
        client_options={"api_endpoint": os.getenv('ENDPOINT_URL')},
        temperature=0
    )

    try:
        response = llm.invoke("What is at the center of most of the galaxies?")
        print(response.text)

        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]  # append-only conversation


        tools = [get_weather]
        llm_with_tools = llm.bind_tools(tools)


        def agent_node(state: AgentState) -> dict:
            """Send the conversation to Gemini and append the reply."""
            response = llm_with_tools.invoke(state["messages"])

            if response.tool_calls:
                print(f"[Agent] Calling: {[c['name'] for c in response.tool_calls]}")
            else:
                print("[Agent] Answering directly.")

            return {"messages": [response]}


        print("agent_node defined.")

        from langgraph.graph import StateGraph, START, END
        from langgraph.prebuilt import ToolNode, tools_condition

        graph_builder = StateGraph(AgentState)

        graph_builder.add_node("agent", agent_node)
        graph_builder.add_node("tools", ToolNode(tools))

        graph_builder.add_edge(START, "agent")

        # The agent picks its own next step: run a tool, or finish.
        graph_builder.add_conditional_edges("agent", tools_condition)

        # Tool results go back to the agent. This edge closes the loop.
        graph_builder.add_edge("tools", "agent")

        app = graph_builder.compile()

        print("Graph compiled: START -> agent -> (tools -> agent)* -> END")

        from IPython.display import Image, display

        try:
            display(Image(app.get_graph().draw_mermaid_png()))
        except Exception:
            print(app.get_graph().draw_mermaid())

    except Exception as e:
        print(e)

if __name__ == "__main__":
    # Call it directly to check it works. No agent involved yet.
    agentic()
