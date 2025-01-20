import os
import sys
from agent.agent import Agent
from agent.interactions.conversation import Conversation
from typing import Dict, Any
from providers.llm.azure_openai import AzureOpenAILLMProvider
# from providers.embeddings.azure_openai import AzureOpenAIEmbeddingProvider
from providers.tools.kubernetes_pod import KubernetesPodTool
from providers.connections.kubernetes import KubernetesConnection
from providers.tools.kubernetes_logs import KubernetesLogsTool
# from agent.vector_db.chroma_db import ChromaVectorDB
# from agent.data_sources.text_file import TextFileDataSource
from pprint import pprint

def load_config() -> Dict[str, Any]:
    """Load the configuration for the agent."""
    kubernetes_conn = KubernetesConnection.get_connection()
    
    return {
        "COGNITIVE_ENGINE": {
            "LLM_PROVIDER": AzureOpenAILLMProvider,
            "LLM_PROVIDER_KWARGS": {
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "deployment_name": "gpt-4o-default"
            }
        },
        "RETRIEVER": {
            "ENABLED": False,
            # "NUM_RELEVANT_DOCUMENTS": 5,
            # "EMBEDDING_PROVIDER": AzureOpenAIEmbeddingProvider,
            # "EMBEDDING_PROVIDER_KWARGS": {
            #     "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            #     "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            #     "deployment_name": "text-embedding-3-small"
            # },
            # "VECTOR_DB": ChromaVectorDB,
            # "VECTOR_DB_KWARGS": {
            #     "persist_directory": "chroma_db_data"
            # },
            # "DATA_SOURCES": {
            #     "kubectl_debug": TextFileDataSource(base_directory="providers/data/kubectl_debug.txt")
            # }
        },
        "TOOL_MANAGER": {
            "ENABLED": True,
            "TOOLS": {
                "kubernetes_pod": {
                    "TOOL_CLASS": KubernetesPodTool,
                    "TOOL_KWARGS": {
                        "kubernetes_connection": kubernetes_conn
                    }
                },
                "kubernetes_logs": {
                    "TOOL_CLASS": KubernetesLogsTool,
                    "TOOL_KWARGS": {
                        "kubernetes_connection": kubernetes_conn
                    }
                }
            }
        }
    }

def main():
    # Initialize the agent with configuration
    config = load_config()
    agent = Agent(config_dict=config)
    
    # Set up the retrieval system
    agent.setup()

    conversation = Conversation(agent)

    
    print("Agent initialized. Type 'quit' or 'exit' to end the conversation.")
    print("\nYou: ", end='', flush=True)

    # Read from stdin line by line
    for line in sys.stdin:
        user_input = line.strip()
        
        if user_input.lower() in ['quit', 'exit']:
            break
        elif not user_input:
            print("\nYou: ", end='', flush=True)
            continue

        # Get response from agent
        response = agent.respond(user_input)
        print("\nMessage:")
        pprint(response)
        print("\nYou: ", end='', flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except EOFError:
        print("\nInput stream closed. Exiting...")
    sys.exit(0)
