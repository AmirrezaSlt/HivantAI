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
        self.tools = toolkit if toolkit else {}
        self.toolkit = toolkit
    
    def set_toolkit(self, toolkit):
        """
        Set the toolkit for this prompt template.
        
        Args:
            toolkit: Toolkit instance containing available tools
        """
        self.toolkit = toolkit
        self.toolkit = toolkit
        
    def render(self):
        """Render the template with the current values."""
        template_str = """
        Your name is {{ name }}, you are an AI agent charged with:
        {{ role }}
        You are given the following permissions:
        {{ permissions }}
        {% if toolkit %}
        You have access to the following tools:
        {% for tool_id, tool in toolkit.tools.items() %}
        - tool_id: "{{ tool_id }}"
            description: {{ tool.description }}
        {% endfor %}
        {% endif %}

        {{ response_format_prompt }}
        """
        
        template = Template(template_str)
        return template.render(
            name=self.name,
            role=self.role,
            permissions=self.permissions,
            toolkit=self.toolkit,
            response_format_prompt=RESPONSE_FORMAT_PROMPT
        )
    
    @property
    def system_prompt(self) -> str:
        return self.render()