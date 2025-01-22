system_prompt_template = """
    You are an AI assistant. Please provide your responses in the following JSON format, without any additional formatting or code block markers:

    For a final response:
    {"response_type": "final", "content": "Your response here"}

    For clarification:
    {"response_type": "clarification", "question": "Your clarification question here"}

    {% if tools %}
    For tool usage:
    {"response_type": "tool_use", "tool_name": "name_of_the_tool", "tool_input": {"param1": "value1", "param2": "value2"}}
    {% endif %}

    Always use this exact format for your responses, with no additional text or formatting before or after the JSON.

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

context_prompt_template = """
    {% if relevant_documents %}
    Try to use the following documents to answer the user's question:
    {% for document in relevant_documents %}
    - {{ document.id }}: {{ document.get_data() }}
    {% endfor %}
    {% endif %}


    {% if reasoning_steps.trail %}
    Relevant context:
    {% for entry in reasoning_steps.trail %}
    - {{ entry.type }}: {{ entry.title }}
        {% if entry.data %}
        {% for key, value in entry.data.items() %}
            {{ key }}: {{ value }}
        {% endfor %}
        {% endif %}
    {% endfor %}
    {% endif %}
"""
