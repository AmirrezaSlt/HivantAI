from typing import Dict, Any
from .cognitive_engine import CognitiveEngine
from .retriever import Retriever
from .toolkit import Toolkit

class Agent:
    def __init__(self, toolkit: Toolkit, cognitive_engine: CognitiveEngine, retriever: Retriever):
        self.toolkit = toolkit
        self.cognitive_engine = cognitive_engine
        self.retriever = retriever
    
    def setup(self):
        if self.retriever:
            self.retriever.setup()
    
    def load_data_to_vector_db(self):
        if self.retriever:
            self.retriever.load_data_to_vector_db()
      
    def respond(self, user_message: str) -> str:
        relevant_documents = self.retriever.query_and_retrieve(query=user_message) if self.retriever else None
        available_tools = self.toolkit.tools if self.toolkit else None
        
        response = self.cognitive_engine.respond(
            user_message=user_message, 
            relevant_documents=relevant_documents,
            tools=available_tools
        )
        return response
