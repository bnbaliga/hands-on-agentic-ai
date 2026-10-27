#Customer service systems

from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types
from pydantic import BaseModel, Field
from typing import Dict, Any

class TicketOutput(BaseModel):
    ticket: Dict[str, Any] = Field(description="Ticket details")
    priority: str = Field(description="Priority level")

customer_agent = LlmAgent(
    model="gemini-flash-latest",
    name="customer_service_agent",
    description="Handles customer intake, classification, and ticketing",
    instruction="[Detailed persona with boundaries]",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)
    ),
    output_schema=TicketOutput
)

#Data processing pipelines

from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class AnalysisOutput(BaseModel):
    insights: List[str] = Field(description="Key insights")
    metrics: Dict[str, Any] = Field(description="Performance metrics")

analysis_agent = LlmAgent(
    model="gemini-pro-latest",
    name="data_processing_agent",
    description="Analyzes input datasets and produces structured metric outputs",
    instruction="[Analysis framework]",
    output_schema=AnalysisOutput
)

#Workflow Automation

from google.adk.agents import LlmAgent
from google.genai import types

workflow_agent = LlmAgent(
    model="gemini-flash-latest",
    name="approval_workflow_agent",
    description="Automates policy checking and approval routing",
    instruction="[Process rules]",
    output_key="approved",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1,# Deterministic for compliance
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
            )
        ]
    )
)


#Educational Assists

from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

tutor_agent = LlmAgent(
    model="gemini-pro-latest",
    name="educational_tutor_agent",
    description="Provides guided step-by-step tutoring using deliberate pedagogical planning",
    instruction="[Teaching methodology]",
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=2048)
    ),
    generate_content_config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=4000
    ),
)


#Code patterns reference
#Pattern 1: Professional service agent

from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types
from pydantic import BaseModel, Field
from typing import Dict, Any

class ServiceResponse(BaseModel):
    response: str = Field(description="The response to the user")
    action_required: bool = Field(description="Whether action is needed")
    metadata: Dict[str, Any] = Field(default=dict, description="Additional metadata")

professional_agent = LlmAgent(
    model="gemini-flash-latest",
    name="service_pro",
    description="Professional service agent handling customer requests with structured output",
    instruction="""
    # PERSONA
    You are [Name], a [Role] with [Experience].

    # BOUNDARIES
    Never: [List of restrictions]
    Always: [List of requirements]

    # EXAMPLES
    [Few-shot examples]
    """,
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_budget=1024)
    ),
    output_schema=ServiceResponse,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=2500,
        safety_settings=[
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
            )
        ]
    )
)


#Pattern 2: Data extraction pipeline
from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from google.genai import types

class Entity(BaseModel):
    type: str = Field(description="Entity type")
    value: str = Field(description="Entity value")
    confidence: float = Field(description="Confidence score 0-1")

class ExtractionOutput(BaseModel):
    entities: List[Entity] = Field(description="Extracted entities")
    relationships: List[str] = Field(description="Relationships found")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extraction metadata")

extractor = LlmAgent(
    model="gemini-flash-latest",
    name="data_extractor",
    description="Extracts structured entities and relationships from raw text",
    instruction="Extract structured data from text",
    output_schema=ExtractionOutput,
    generate_content_config=types.GenerateContentConfig(
       temperature=0.1, # Low for deterministic extraction
       thinking_config=types.ThinkingConfig(
       thinking_budget=0
       ), #  Preserve entire token budget for output
    max_output_tokens=1000,
    ),
)

#Pattern 3: Decision making system
from google.adk.agents import LlmAgent
from google.adk.planners import BuiltInPlanner
from google.genai import types

decision_agent = LlmAgent(
    model="gemini-pro-latest",
    name="decision_maker",
    description="Evaluates requests against organizational policies",
    instruction="""
    Evaluate requests against policy.
    Consider all factors and dependencies.
    """,
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(include_thoughts=False, thinking_budget=2048)
    ),
    output_key="approved",  # Store result in session state
    generate_content_config=types.GenerateContentConfig(
        temperature=0.1, # strict consistency for policy evaluation
        max_output_tokens=3000,
    ),
)