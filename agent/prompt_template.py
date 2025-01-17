from jinja2 import Template
from typing import List, Dict

class SystemPrompt:
    def __init__(self):
        self.prompt = """
        You are an AI assistant. Please provide your responses in the following JSON format, without any additional formatting or code block markers:

        For a final response:
        {"response_type": "final", "content": "Your response here"}

        For tool usage:
        {"response_type": "tool_use", "tool_name": "name_of_the_tool", "tool_input": {"param1": "value1", "param2": "value2"}}

        For clarification:
        {"response_type": "clarification", "question": "Your clarification question here"}

        Always use this exact format for your responses, with no additional text or formatting before or after the JSON.
        """

    def update(self, new_prompt: str):
        self.prompt = new_prompt

class ContextPrompt:
    def __init__(self, template: str = """
        {% if cognitive_trail.trail %}
        Relevant context:
        {% for entry in cognitive_trail.trail %}
        - {{ entry.type }}: {{ entry.title }}
        {% if entry.data %}
          {% for key, value in entry.data.items() %}
            {{ key }}: {{ value }}
          {% endfor %}
        {% endif %}
        {% endfor %}
        {% endif %}
        {% if tools %}
        These are the tools you have access to, try to utilize them when necessary:
        {% for name, tool in tools.items() %}
            - {{ name }}: {{ tool.description }}
            Inputs: {{ tool.info.inputs }}
            Outputs: {{ tool.info.outputs }}
        {% endfor %}
        To use a tool, respond with a JSON object in the following format:
        {"response_type": "tool_use", "tool_name": "tool_name", "tool_input": {"param1": "value1", "param2": "value2"}}
        Only use tools when necessary. If you don't need to use a tool, respond normally.
        {% endif %}
        """
    ):
        self.template = Template(template)

    def update(self, new_template: str):
        self.template = Template(new_template)

    def render(self, cognitive_trail, tools: List[Dict]) -> List[Dict[str, str]]:
        context = self.template.render(cognitive_trail=cognitive_trail, tools=tools)
        messages = [{"role": "system", "content": context}]

        for entry in cognitive_trail.trail:
            if entry.type == "received_message":
                messages.append({"role": "user", "content": entry.data["message"]})
            elif entry.type == "final_response":
                messages.append({"role": "assistant", "content": entry.data["content"]})
            elif entry.type in ["used_tool", "relevant_documents"]:
                tool_message = f"{entry.type.capitalize()}: {entry.title}\n"
                for key, value in entry.data.items():
                    tool_message += f"{key}: {value}\n"
                messages.append({"role": "system", "content": tool_message})

        return messages