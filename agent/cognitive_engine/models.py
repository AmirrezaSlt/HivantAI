from typing import List, Dict, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4

class ReasoningStepType(Enum):
    RECEIVED_MESSAGE = "received_message"
    USED_TOOL = "used_tool"
    FINAL_RESPONSE = "final_response"
    CLARIFICATION_REQUEST = "clarification_request"
    
class ReasoningStep(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    type: ReasoningStepType
    title: str
    data: Dict[str, Any] = Field(default_factory=dict)
    status: str = "success"

class ReasoningSteps(BaseModel):
    trail: List[ReasoningStep] = Field(default_factory=list)

    def add_entry(self, type: ReasoningStepType, title: str, data: Dict[str, Any] = None, status: str = "success"):
        entry = ReasoningStep(
            type=type,
            title=title,
            data=data or {},
            status=status
        )
        self.trail.append(entry)
