from typing import List, Dict, Any, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_serializer
from agent.input import Input
from agent.toolkit.tool import BaseTool
from agent.retriever.reference_documents import BaseReferenceDocument

class ReasoningStepType(str, Enum):
    INPUT = "input"
    Reference_DOCUMENTS = "reference_documents"
    TOOL = "tool"
    RESPONSE = "response"
    
class BaseReasoningStep(BaseModel):
    type: Literal[ReasoningStepType.INPUT, ReasoningStepType.TOOL, ReasoningStepType.RESPONSE]
    title: str = Field(
        description="Brief, human-readable description of the step",
    )

    explanation: Optional[str] = Field(
        default=None,
        description="Optional explanation of why this step was chosen",
    )

    model_config = {
        "discriminator_key": "type"
    }

class ToolStep(BaseReasoningStep):
    type: Literal[ReasoningStepType.TOOL] = ReasoningStepType.TOOL
    tool_name: str = Field(description="Name of the tool to use")
    input_data: Dict[str, Any] = Field(description="Input data for the tool")
    output_data: Dict[str, Any] = Field(description="Output data from the tool")

class ResponseStep(BaseReasoningStep):
    type: Literal[ReasoningStepType.RESPONSE] = ReasoningStepType.RESPONSE
    response: str = Field(description="Response to the user")

class InputStep(BaseReasoningStep):
    type: Literal[ReasoningStepType.INPUT] = ReasoningStepType.INPUT
    input: Input = Field(description="Input of the user")

class RelevantDocumentsStep(BaseReasoningStep):
    type: Literal[ReasoningStepType.Reference_DOCUMENTS] = ReasoningStepType.Reference_DOCUMENTS
    reference_documents: List[BaseReferenceDocument] = Field(default_factory=list)

    model_config = {
        "arbitrary_types_allowed": True
    }

    @field_serializer('reference_documents')
    def serialize_documents(self, docs: List[BaseReferenceDocument]) -> List[Dict[str, Any]]:
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "content": doc.content,
            } for doc in docs
        ]

class LLMResponse(BaseModel):
    step: Union[ToolStep, ResponseStep] = Field(
        description="Reasoning step for the LLM response",
        discriminator='type'
    )

class ReasoningState:
    
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools
        self.trail = []

    def update_state_from_llm_response(self, response: LLMResponse) -> None:
        """Update the reasoning state from a raw LLM response string."""
        self.trail.append(response.step)

    def update_state_from_input(self, input: Input, reference_documents: List[BaseReferenceDocument]) -> None:
        self.trail.append(InputStep(
            title="User Input",
            input=input
        ))
        if reference_documents:
            self.trail.append(RelevantDocumentsStep(
                title="Retrieved relevant documents",
                reference_documents=reference_documents
            ))
    
    @property
    def system_prompt(self) -> str:
        prompt = [
            "You are an AI assistant that thinks step by step.",
            "",
            "Format your response using the following schema:",
            str(LLMResponse.model_json_schema()),
            ""
        ]
        
        if self.tools:
            prompt.extend([
                "Available tools:",
                *[f"- {tool.id}: {tool.description}" for tool in self.tools],
                "  Input schema: {tool.input_schema}",
                "  Output schema: {tool.output_schema}",
                ""
            ])
        return "\n".join(prompt)
    
    @property
    def messages(self) -> List[Dict[str, str]]:
        """Returns the message list for the LLM with the current context"""
        messages = [{"role": "system", "content": self.system_prompt}]
        for step in self.trail:
            content = step.model_dump_json()
            if isinstance(step, InputStep):
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "assistant", "content": content})
        return messages
