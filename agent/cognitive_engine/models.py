from typing import List, Dict, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import uuid4

class CognitiveEventType(Enum):
    RECEIVED_MESSAGE = "received_message"
    RELEVANT_DOCUMENTS = "relevant_documents"
    USED_TOOL = "used_tool"
    FINAL_RESPONSE = "final_response"
    CLARIFICATION_REQUEST = "clarification_request"
    
class CognitiveTrailEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.utcnow())
    type: CognitiveEventType
    title: str
    data: Dict[str, Any] = Field(default_factory=dict)
    status: str = "success"

class CognitiveTrail(BaseModel):
    trail: List[CognitiveTrailEntry] = Field(default_factory=list)

    def add_entry(self, type: CognitiveEventType, title: str, data: Dict[str, Any] = None, status: str = "success"):
        entry = CognitiveTrailEntry(
            type=type,
            title=title,
            data=data or {},
            status=status
        )
        self.trail.append(entry)

class ConversationEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    role: str
    content: str

class ConversationHistory(BaseModel):
    messages: List[ConversationEntry] = Field(default_factory=list)

    def add_entry(self, role: str, content: str):
        entry = ConversationEntry(role=role, content=content)
        self.messages.append(entry)
