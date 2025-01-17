from typing import Dict, Any, List, Tuple
from pydantic import BaseModel
from .cognitive_engine import CognitiveEngine
from .cognitive_engine.models import CognitiveTrail
from .retriever import Retriever
from .tool_manager import ToolManager
from .config import Config

class AgentResponse(BaseModel):
    message: str
    relevant_documents: List[Any]
    cognitive_trail: CognitiveTrail

class Agent:
    def __init__(self, config_dict: Dict[str, Any] = None):
        self.config = Config(**(config_dict or {}))
        self.tool_manager = ToolManager(config=self.config) if self.config.TOOL_MANAGER.ENABLED else None
        self.cognitive_engine = CognitiveEngine(
            config=self.config,
            tools=self.tool_manager.tools if self.tool_manager else None
        )
        self.retriever = Retriever(config=self.config) if self.config.RETRIEVER.ENABLED else None
    
    def setup(self):
        if self.retriever:
            self.retriever.setup()
      
    def respond(self, user_message: str) -> AgentResponse:
        relevant_documents = []
        if self.retriever:
            relevant_documents = self.retriever.query_and_retrieve(
                query=user_message
            )
        
        response, cognitive_trail = self.cognitive_engine.respond(
            user_message=user_message, 
            relevant_documents=relevant_documents
        )
        
        return AgentResponse(
            message=response,
            relevant_documents=relevant_documents,
            cognitive_trail=cognitive_trail
        )

    def reset(self):
        self.cognitive_engine.reset()

    def load_data(self, retriever_uri_pairs: List[Tuple[str, str]]):
        if self.retriever:
            self.retriever.load_content_to_vector_db(retriever_uri_pairs)