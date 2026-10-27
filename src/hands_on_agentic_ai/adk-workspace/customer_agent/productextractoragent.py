# """
# Product extraction agent with structured JSON output.
# Demonstrates ADK's output_schema with Pydantic BaseModel on Gemini platform.
# """
#
# from google.adk.agents import LlmAgent
# from pydantic import BaseModel, Field
# from google.genai import types
# from google.adk.planners import BuiltInPlanner
#
# # Step 1: Define the output structure with Pydantic
# # class ProductInfo(BaseModel):
# #     product_name: str = Field(description="The full name of the product")
# #     price: float = Field(description="The price in USD")
# #     storage: str = Field(description="Storage capacity (e.g., '256GB')")
# #     color: str = Field(default="Not specified", description="Product color if mentioned")
# #
# # # Step 2: Create agent with output_schema
# # root_agent = LlmAgent(
# #     model="gemini-flash-latest",
# #     name="product_extractor",
# #     description="Extracts product information from user messages and returns structured JSON",
# #     instruction="""You are a Product Information Extractor.
# #
# # Your task:
# # - Read the user's message about a product
# # - Extract: product_name, price, storage, and color (if mentioned)
# # - Respond ONLY with valid JSON matching this format:
# #
# # {
# #   "product_name": "product name here",
# #   "price": 999.99,
# #   "storage": "256GB",
# #   "color": "Space Black"
# # }
# #
# # Rules:
# # - price must be a number (no dollar signs)
# # - storage must include unit (GB, TB)
# # - If color not mentioned, use "Not specified"
# # - Output ONLY the JSON, no explanation text""",
# #     output_schema=ProductInfo,  # Enforce this exact structure
# #     output_key="extracted_product"  # Store result in session state
# # )
#
#
# root_agent = LlmAgent(
#     model="gemini-flash-light",
#     name="configured_agent",
#     description="Agent configured with customized generation parameters",
#     generate_content_config=types.GenerateContentConfig(
#         temperature=0.2,          # More deterministic output
#         max_output_tokens=250,    # Limit response length
#         top_p=0.8,               # Nucleus sampling
#         top_k=10                 # Top-k sampling
#     )
# )
# #
# # planning_agent = LlmAgent(
# #     model="gemini-flash-latest",
# #     name="systematic_planner",
# #     description="Solves complex multi-step reasoning problems using native thinking",
# #     planner=BuiltInPlanner(
# #         thinking_config=types.ThinkingConfig(
# #             include_thoughts=True,   # Show reasoning in response
# #             thinking_budget=1024     # Tokens allocated for thinking
# #         )
# #     ),
# #     instruction="Solve complex problems systematically"
# # )