import os

# Import agent components
from agent.agent import Agent
from agent.cognitive_engine import CognitiveEngine
from agent.retriever import Retriever
from agent.toolkit import Toolkit
from agent.toolkit.config import PythonCodeExecutorConfig

# Import providers
from providers.llm.azure_openai import AzureOpenAILLMProvider
from providers.embeddings.azure_openai import AzureOpenAIEmbeddingProvider

def create_agent():
    """
    Create and initialize the agent with configurations.
    
    Returns:
        Agent: The initialized agent instance
    """
    cognitive_engine = CognitiveEngine(
        SYSTEM_PROMPT="""
        You are an AI assistant that tries to help the user with their Kubernetes problems. 
        You already have access to the main kubernetes cluster by default so assume that's the cluster you're working with and prefer to get data yourself than asking the user for it if possible. 
        You can use the code execution tool to run python code that uses the kubernetes client to run kubernetes commands and get the output.
        Try to do incremental steps and get to a good response and feel free to use the tools multiple times, the previous steps taken will be provided to you. 
        Keep your codes small and atomic and try to debug through multiple steps rather than one large block of code.
        """,
        LLM_PROVIDER=AzureOpenAILLMProvider(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4o-default"
        ),
        MAX_ITERATIONS=10,
        AGENT_NAME="Kubernetes Agent",
        AGENT_ROLE="You are an AI assistant that tries to help the user with their Kubernetes problems.",
        AGENT_PERMISSIONS=["kubernetes"]
    )

    toolkit = Toolkit(
        TOOLS=[],
        EXECUTOR=PythonCodeExecutorConfig(
            base_image="python:3.13.1-slim",
            python_version="3.13",
            python_packages=["kubernetes==31.0.0"],
            environment_variables={
                "PYTHONUNBUFFERED": {
                    "value": "1",
                },
                "PYTHONDONTWRITEBYTECODE": {
                    "value": "1",
                }
            },
            resource_requests={"cpu": "200m", "memory": "512Mi"},
            resource_limits={"cpu": "200m", "memory": "512Mi"}
        )
    )

    agent = Agent(
        cognitive_engine=cognitive_engine,
        retriever=None,
        toolkit=toolkit
    )
    agent.setup()
    # Uncomment if you want to load vector DB data on startup
    # agent.load_data_to_vector_db()
    
    return agent

def main():
    """
    Main entry point for the application.
    Creates the agent and starts the server.
    """
    # Create and initialize agent
    agent = create_agent()
    print("Agent initialized successfully")
    
    # Start the server
    agent.start_server(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main() 