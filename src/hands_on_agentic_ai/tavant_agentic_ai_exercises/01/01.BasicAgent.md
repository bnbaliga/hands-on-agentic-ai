Building a Single Agent with LangGraph and Google Gemini
1. Introduction
An LLM on its own cannot tell you the weather. It has a training cutoff and no network access, so anything happening right now is out of reach.

An agent solves this. We give the model a weather tool and let it decide when to use one. This lab builds that agent with LangGraph in about forty lines of code:

Question -> Agent -> (calls tool) -> Weather API -> Agent -> Answer
One model, one tool, one loop.

2. Problem Statement
The agent should handle both of these correctly, without us writing any routing logic:

Question	Expected behaviour
"What is the weather in Tokyo right now?"	Call the weather tool, then answer from the result
"Why is the sky blue?"	Answer directly, no tool needed
Nobody tells the agent which is which. We describe the tool, and the model decides.

That requires three pieces:

A tool the model can call, described well enough that it knows when it applies.
A loop, so the model can call the tool, read the result, and then answer.
A router, to decide after each model reply whether to run a tool or stop.
3. How It Works
The graph has two nodes and one branch:

agent sends the conversation to Gemini, which replies with either a final answer or a tool call.
tools runs any requested tool and appends the result to the conversation.
tools_condition inspects the agent's last message: a tool call means go to tools, otherwise stop.
[START] -> agent --(tool call)--> tools
             ^                      |
             |______________________|

           agent --(answer)--> [END]
The edge from tools back to agent is the whole trick. It turns a single model call into a loop: think, act, observe, answer. A weather question runs the agent node twice, once to request the tool and once to interpret what came back.

Only agent reasons. The tools node just runs a Python function, which is why this is still a single agent despite having two nodes.

4. Input and Output
Input is a plain string passed to app.invoke(...) as a user message.

Output is the final message in the returned state. Every intermediate step is kept too, so the full trail is inspectable: the question, the tool call, the raw tool result, and the answer built on it.

5. Pre-requisites
Basic Python and familiarity with running notebook cells.
Your platform API key and endpoint URL, pasted directly into the credentials cell below.
The weather data comes from Open-Meteo, which needs no key or account.

6. Environment Setup
langgraph provides the graph, the prebuilt ToolNode, and the tools_condition router. langchain-google-genai is the Gemini wrapper. requests calls the weather API.


%pip install -q langgraph langchain-google-genai langchain-core requests
Step 1: Configure platform credentials
Credentials are set directly as variables here: GOOGLE_API_KEY is your platform token, and ENDPOINT_URL is the platform's Gemini proxy endpoint. Both are passed straight into ChatGoogleGenerativeAI when the model is initialized in the next step.


# ---- Platform Credentials ---------------------------------------------------
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
ENDPOINT_URL   = "YOUR_ENDPOINT"

print(f"Endpoint : {ENDPOINT_URL}")
print(f"API key  : configured")
Step 2: Initialize the model
ChatGoogleGenerativeAI takes the credentials directly via google_api_key and client_options, so no separate configuration call is needed. temperature=0 keeps the agent's tool decisions consistent between runs.


from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=GOOGLE_API_KEY,
    client_options={"api_endpoint": ENDPOINT_URL},
    temperature=0,
)

print("Model ready:", llm.model)
Step 3: Build the weather tool
The @tool decorator turns a normal function into something the model can call. The docstring matters: it is sent to the model as the tool's description and is the only thing it uses to decide when the tool applies.

Open-Meteo needs coordinates rather than a place name, so the tool geocodes the city first, then fetches current conditions. Errors are returned as strings rather than raised, letting the agent report the problem instead of crashing the graph.


import requests
from langchain_core.tools import tool

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


tools = [get_weather]

# Call it directly to check it works. No agent involved yet.
print(get_weather.invoke({"city": "Tokyo"}))
Step 4: Define the shared state
The state is just the conversation. add_messages is a reducer: it tells LangGraph to append what each node returns rather than replace the list, which is what lets the tool result join the conversation instead of wiping it.


from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # append-only conversation


print("AgentState defined.")
Step 5: Define the agent node
llm.bind_tools(tools) attaches the tool's name, description, and argument schema to the model. Gemini can then reply with a structured tool_calls field instead of prose, and the node simply appends whatever comes back.


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
Step 6: Assemble the graph
Two lines do the real work: add_conditional_edges lets the agent choose its next step, and add_edge("tools", "agent") sends tool results back for another pass, closing the loop.


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
Step 7: Visualize the graph
The dotted lines out of agent are the conditional edges; the solid line from tools back to agent is the cycle.


from IPython.display import Image, display

try:
    display(Image(app.get_graph().draw_mermaid_png()))
except Exception:
    print(app.get_graph().draw_mermaid())
Step 8: Run the agent
Two questions, two different paths through the same graph. Watch the [Agent] lines: the weather question should print a tool call and then a final answer, which is the loop running twice.


# 1. Needs live data -> should call the weather tool
result = app.invoke({"messages": [("user", "What is the weather in Tokyo right now?")]})

msg = result["messages"][-1]
print("\n", msg.content if isinstance(msg.content, str) else "".join(p.get("text", "") for p in msg.content))

# 2. No tool needed -> should answer directly
result = app.invoke({"messages": [("user", "Why is the sky blue?")]})

msg = result["messages"][-1]
print("\n", msg.content if isinstance(msg.content, str) else "".join(p.get("text", "") for p in msg.content))
Step 9: Inspect the reasoning trail
The final answer hides the steps that produced it. Printing every message shows the full loop: question, tool call, raw tool result, answer.


result = app.invoke({"messages": [("user", "How warm is it in Paris?")]})

print(f"\n{'=' * 60}")
for msg in result["messages"]:
    kind = msg.__class__.__name__.replace("Message", "").upper()
    print(f"\n[{kind}]")

    if getattr(msg, "tool_calls", None):
        for call in msg.tool_calls:
            print(f"  -> {call['name']}({call['args']})")

    text = msg.content if isinstance(msg.content, str) else "".join(p.get("text", "") for p in msg.content)
    if text.strip():
        print(f"  {text[:300]}")
7. What We Learnt
We built a single tool-calling agent as a cyclic LangGraph. One StateGraph held an append-only message list, an agent node sent it to Gemini with a tool bound to it, a prebuilt ToolNode ran whatever Gemini asked for, and tools_condition decided each pass whether to loop or stop.

add_messages appends to state instead of overwriting, so the conversation accumulates
bind_tools hands the model a tool schema so it can request calls itself, with no routing code from us
The docstring is the model's only guide to when a tool applies
ToolNode and tools_condition are prebuilt execution and routing for the agent loop
The tools -> agent back edge is what makes the graph cyclic, and therefore an agent