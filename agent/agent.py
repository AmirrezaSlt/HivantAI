from typing import Dict, Any
from .cognitive_engine import CognitiveEngine
from .retriever import Retriever
from .tool_manager import ToolManager
from .config import Config

class Agent:
    def __init__(self, config_dict: Dict[str, Any] = None):
        self.config = Config(**(config_dict or {}))
        self.tool_manager = ToolManager(config=self.config) if self.config.TOOL_MANAGER.ENABLED else None
        self.cognitive_engine = CognitiveEngine(
            config=self.config,
        )
        self.retriever = Retriever(config=self.config) if self.config.RETRIEVER.ENABLED else None
    
    def setup(self):
        if self.retriever:
            self.retriever.setup()
    
    def load_data_to_vector_db(self):
        if self.retriever:
            self.retriever.load_data_to_vector_db()
      
    def respond(self, user_message: str) -> str:
        relevant_documents = self.retriever.query_and_retrieve(query=user_message) if self.retriever else None
        available_tools = self.tool_manager.tools if self.tool_manager else None
        
        response = self.cognitive_engine.respond(
            user_message=user_message, 
            relevant_documents=relevant_documents,
            tools=available_tools
        )
        return response
