from agent.agent import Agent
from typing import List
from dataclasses import dataclass

@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str

class Conversation:
    def __init__(self, agent: Agent):
        self.messages: List[Message] = []
        self.agent = agent

    @classmethod
    def resume(cls, agent: Agent, messages: List[Message]) -> "Conversation":
        """Resume a conversation with existing messages.
        
        Args:
            agent: The agent to use for the conversation
            messages: List of previous messages
                           
        Returns:
            A new Conversation instance with the loaded messages
        """
        conversation = cls(agent)
        conversation.messages = messages
        return conversation

    def send_message(self, message: str) -> str:
        # Add user message
        self.messages.append(Message(role="user", content=message))
        
        # Get response from agent
        response = self.agent.respond(message)
        
        # Add assistant message
        self.messages.append(Message(role="assistant", content=response))
        
        return response
