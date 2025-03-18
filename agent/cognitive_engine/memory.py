from typing import List

class Conversation:
    def __init__(self):
        self.messages = []

    def add_message(self, role: str, message: str):
        self.messages.append({"role": role, "content": message})


class Memory:
    """
    Contains the conversation history.
    """

    def __init__(self):
        self.conversation = Conversation()
  