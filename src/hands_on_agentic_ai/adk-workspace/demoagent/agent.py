#
#
# # root_agent = Agent(
# #     model='gemini-3.5-flash',
# #     name='root_agent',
# #     description='A helpful assistant for user questions.',
# #     instruction='Answer user questions to the best of your knowledge',
# # )
#
# from google.adk.agents.llm_agent import Agent
# from google.adk.agents import LlmAgent
# from google.adk.planners import BuiltInPlanner
# from google.adk.planners import PlanReActPlanner
# from google.genai import types
# import os
#
# root_agent = LlmAgent(
#     # 2. Swap out the model name to an OpenAI model
#     model="gpt-4o",
#     name="systematic_planner",
#     description="Solves complex multi-step reasoning problems systematically",
#     instruction="Solve complex problems systematically",
#     planner=PlanReActPlanner()
#     # planner=BuiltInPlanner(
#     #         thinking_config=types.ThinkingConfig(
#     #             include_thoughts=True,   # Show reasoning in response
#     #             thinking_budget=1024     # Tokens allocated for thinking
#     #         )
#     #     ),
#     # Note: `ThinkingConfig` and `types.ThinkingConfig` are native to Gemini 3.5/3.6.
#     # For OpenAI reasoning models (like o1-preview or o3-mini), reasoning token allocation
#     # is handled automatically or via alternative parameters depending on ADK's OpenAI connector.
# )
#
# # root_agent = LlmAgent(
# #     model="gemini-3.5-flash",
# #     name="systematic_planner",
# #     description="Solves complex multi-step reasoning problems using native thinking",
# #     # planner=BuiltInPlanner(
# #     #     thinking_config=types.ThinkingConfig(
# #     #         include_thoughts=True,   # Show reasoning in response
# #     #         thinking_budget=1024     # Tokens allocated for thinking
# #     #     )
# #     # ),
# #     instruction="Solve complex problems systematically"
# # )


"""
Problem-solving agent with built-in planning capabilities.
Demonstrates ADK's BuiltInPlanner with ThinkingConfig.
"""

from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

# Planning-enabled agent for complex problem solving
root_agent = LlmAgent(
    model="gemini-3.5-flash",
    name="strategic_problem_solver",
    description="Solves complex problems using multi-step reasoning and planning",
    instruction="""You are a Strategic Problem Solver.

Your approach to complex problems:
1. **Understand** - Break down the problem into components
2. **Analyze** - Consider multiple approaches and trade-offs
3. **Plan** - Develop a step-by-step solution strategy
4. **Execute** - Provide clear, actionable recommendations

For complex problems:
- Think through implications and edge cases
- Consider short-term vs long-term consequences
- Identify potential risks and mitigation strategies
- Provide reasoning for your recommendations

Be thorough, analytical, and systematic in your approach.""",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,   # Show reasoning process
            thinking_budget=2048     # Large budget for complex thinking
        )
    )
)