from .cognitive_engine import CognitiveEngine
from .retriever import Retriever
from .toolkit import Toolkit
from .input import Input

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
      
    def respond(self, input: Input, state = None):
        """
        Process input and generate a response using the cognitive engine.
        
        Args:
            input: The input message and any attachments
            state: Optional state from previous interactions
            
        Returns:
            When not streaming: A tuple containing (response_text, updated_state)
            When streaming: A generator yielding response chunks
        """
        
        # Pass through the generator from cognitive_engine.respond
        return self.cognitive_engine.respond(
            input=input,
            toolkit=self.toolkit,
            state=state
        )
