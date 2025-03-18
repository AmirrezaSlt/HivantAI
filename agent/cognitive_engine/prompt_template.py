import json
from jinja2 import Template
from .response_parser import RESPONSE_FORMAT_PROMPT
from agent.toolkit import Toolkit
from agent.toolkit.tool import ToolInfo
from typing import List, Optional

class PromptTemplate:
    """Manages prompt templates for interaction with the LLM."""

    def __init__(self, name, role, permissions, toolkit = None):
        self.name = name
        self.role = role
        self.permissions = permissions
        self.tools = toolkit.info if toolkit else {}
        self.toolkit = toolkit
    
    def set_toolkit(self, toolkit):
        """
        Set the toolkit for this prompt template.
        
        Args:
            toolkit: Toolkit instance containing available tools
        """
        self.tools = toolkit.info if toolkit else {}
        self.toolkit = toolkit
        
    def add_message(self, role, content):
        """
        This method is called by CognitiveEngine to track messages.
        In this implementation, it does nothing as we don't need to store 
        these messages in the prompt template.
        """
        pass

    def render(self):
        """Render the template with the current values."""
        template_str = """
      Your name is {{ name }}, you are an AI agent charged with:
      {{ role }}
      You are given the following permissions:
      {{ permissions }}
      {% if tools %}
      You have access to the following tools:
      {% for tool_name, tool_info in tools.items() %}
      - {{ tool_name }}: {{ tool_info.description }}
        {% if tool_info.parameters %}
        Parameters:
        {% for param_name, param_info in tool_info.parameters.items() %}
        - {{ param_name }} ({{ param_info.type }}, {% if param_info.required %}required{% else %}optional{% endif %}): {{ param_info.description }}
        {% endfor %}
        {% endif %}
      {% endfor %}
      {% endif %}

      {{ response_format_prompt }}
      """
        
        template = Template(template_str)
        return template.render(
            name=self.name,
            role=self.role,
            permissions=self.permissions,
            tools=self.tools,
            response_format_prompt=RESPONSE_FORMAT_PROMPT
        )
    
    @property
    def system_prompt(self) -> str:
        return self.render()