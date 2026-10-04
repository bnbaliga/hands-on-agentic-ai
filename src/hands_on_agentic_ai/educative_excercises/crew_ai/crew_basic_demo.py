import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from ddgs import DDGS
from crewai_tools import SerperDevTool
load_dotenv(verbose=True)

#llm = LLM(model="gemini/gemini-3.8-flash", api_key=os.getenv("GEMINI_API_KEY"))

# @tool("Web Search")
# def search_tool(query: str) -> str:
#     """Search the web and return the top results with titles, links and snippets."""
#     try:
#         results = DDGS().text(query, max_results=5)
#     except Exception as e:
#         return f"No search results for '{query}' ({e}). Try a different or broader query."
#     return "\n\n".join(f"{r['title']}\n{r['href']}\n{r['body']}" for r in results)

# Create a search tool
search_tool = SerperDevTool()
#Venue finder agent


venue_finder = Agent(
    role="Conference Venue Finder",
    goal="Find the best venue for the upcoming conference",
    backstory="You are an experienced event planner with a knack for finding the perfect venues. Your expertise ensures that all conference requirements are met efficiently.",
    verbose=True,
)

find_venue_task = Task(
    description="Conduct a thorough search to find the best venue for the {conference_name} "
        "conference with these requirements: {requirements}. Consider factors such as "
        "capacity, location, amenities, and pricing. Use online resources and databases "
        "to gather comprehensive information",
    expected_output="A list of 5 potential venues with detailed information on capacity, location, amenities, pricing, and availability.",
    agent=venue_finder,
    tools=[search_tool],
)

venue_quality_assurance = Agent(
    role="Venue Quality Assurance Specialist",
    goal="Ensure the selected venues meet all quality standards and client requirements",
    backstory=(
        "You are meticulous and detail-oriented, ensuring that the venue options provided "
        "are not only suitable but also exceed the client's expectations. "
        "Your job is to review the venue options and provide detailed feedback."
    ),
    verbose=True,
)

quality_assurance_review_task = Task(
    description=(
        "Review the venue options provided by the Conference Venue Finder for the {conference_name} conference. "
        "Ensure that each venue meets these requirements: {requirements}. "
        "Provide a detailed report on the suitability of each venue."
    ),
    expected_output=(
        "A detailed review of the 5 potential venues, highlighting any issues, strengths, and overall suitability."
    ),
    tools=[search_tool],
    agent=venue_quality_assurance,
)

#assemble the crew

event_planning_crew = Crew(
  agents=[venue_finder, venue_quality_assurance],
  tasks=[find_venue_task, quality_assurance_review_task],
  verbose=True,
  memory=False,  # memory uses OpenAI embeddings by default
  max_rpm=5,  # Gemini free tier allows 5 requests per minute
)
#actual crew execution now
inputs = {
    "conference_name": "AI Innovations Summit",
    "requirements": "Capacity for 5000, central location, modern amenities, budget up to $50,000"
}

result = event_planning_crew.kickoff(inputs=inputs)
print(result)


from crewai import Agent
#
# # Initialize the Gemini model using ChatGoogleGenerativeAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# gemini=ChatGoogleGenerativeAI(model="gemini-1.5-flash",
#                            verbose=True,
#                            temperature=0.5,
#                            google_api_key=os.getenv("GOOGLE_API_KEY"))
#
# # Initialize the GPT-4 model using ChatOpenAI
# from langchain_openai import ChatOpenAI
# gpt=ChatOpenAI(model="gpt-4o-2024-08-06",
#                verbose=True,
#                temperature=0.5,
#                openai_api_key=os.getenv("OPENAI_API_KEY"))
