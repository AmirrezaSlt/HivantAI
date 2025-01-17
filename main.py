import os
import sys
from agent.agent import Agent, AgentResponse
from typing import Dict, Any
from providers.chat.azure_openai import AzureOpenAIChatProvider
from providers.embeddings.openai import OpenAIEmbeddingProvider
from providers.tools.kubernetes_pod import KubernetesPodTool
from providers.connections.kubernetes import KubernetesConnection
from providers.tools.kubernetes_logs import KubernetesLogsTool
from pprint import pprint

def load_config() -> Dict[str, Any]:
    """Load the configuration for the agent."""
    kubernetes_conn = KubernetesConnection.get_connection()
    
    return {
        "COGNITIVE_ENGINE": {
            "CHAT_PROVIDER": AzureOpenAIChatProvider,
            "CHAT_PROVIDER_KWARGS": {
                "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
                "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
                "deployment_name": "gpt-4o-default"
            }
        },
        "RETRIEVER": {
            "ENABLED": False,
            "NUM_RELEVANT_DOCUMENTS": 5,
            "EMBEDDING_PROVIDER": OpenAIEmbeddingProvider,
            "EMBEDDING_PROVIDER_KWARGS": {
                "api_key": os.getenv("OPENAI_API_KEY"),
            }
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
        response: AgentResponse = agent.respond(user_input)
        print("\nMessage:")
        pprint(response.message)
        print("\nCognitive Trail:")
        pprint([trail.title for trail in response.cognitive_trail.trail])
        pprint(response.cognitive_trail)
        if response.relevant_documents:
            print("\nRelevant Documents:")
            pprint(response.relevant_documents)
        print("\nYou: ", end='', flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except EOFError:
        print("\nInput stream closed. Exiting...")
    sys.exit(0)
