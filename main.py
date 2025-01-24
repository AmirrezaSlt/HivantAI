import os
import sys
from agent.agent import Agent
from typing import Dict, Any
from providers.llm.azure_openai import AzureOpenAILLMProvider
from providers.embeddings.azure_openai import AzureOpenAIEmbeddingProvider
from providers.tools.kubernetes_pod import KubernetesPodTool
from providers.connections.kubernetes import KubernetesConnection
from providers.tools.kubernetes_logs import KubernetesLogsTool
# from agent.vector_db.chroma_db import ChromaVectorDB
from providers.vector_dbs.qdrant import QdrantVectorDB
from agent.retriever.reference_documents.text_file import TextFileReferenceDocument
from agent.cognitive_engine import CognitiveEngine
from agent.retriever import Retriever
from agent.toolkit import Toolkit
from agent.agent import Agent
from agent.input import Input

def main():

    kubernetes_conn = KubernetesConnection.get_connection()

    cognitive_engine = CognitiveEngine(
        LLM_PROVIDER=AzureOpenAILLMProvider(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="gpt-4o-default"
        )
    )

    retriever = Retriever(
        NUM_REFERENCE_DOCUMENTS=3,
        EMBEDDING_PROVIDER=AzureOpenAIEmbeddingProvider(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name="text-embedding-3-small",
            dimension=1536
        ),
        VECTOR_DB=QdrantVectorDB(
            dimension=1536,
            in_memory=True,
            host="vector-db",
            port=6333,
            collection="default",
            score_threshold=0.0,
            similarity_metric="Cosine"
        ),
        REFERENCE_DOCUMENTS=[
            TextFileReferenceDocument(id="kubectl_debug_guide", file_path="providers/data/kubectl_debug.txt"),
            TextFileReferenceDocument(id="contact", file_path="providers/data/contact.txt")
        ]
    )
    
    toolkit = Toolkit(
        TOOLS=[
            KubernetesPodTool(
                id="kubernetes_pod",
                kubernetes_connection=kubernetes_conn
            ),
            KubernetesLogsTool(
                id="kubernetes_logs",
                kubernetes_connection=kubernetes_conn
            )
        ]
    )

    agent = Agent(
        cognitive_engine=cognitive_engine,
        retriever=retriever,
        toolkit=toolkit
    )
    agent.setup()
    # agent.load_data_to_vector_db()
    print("Agent initialized. Type 'quit' or 'exit' to end the conversation.")
    print("\nYou: ", end='', flush=True)

    try:
        with open("providers/data/kubectl_access.txt", "rb") as f:
            kubectl_access_content = f.read()
        
        for line in sys.stdin:
            user_input = line.strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
            elif not user_input:
                print("\nYou: ", end='', flush=True)
                continue

            attachments = {"kubectl_access.txt": kubectl_access_content}
            response = agent.respond(
                input=Input(message=user_input, attachments=attachments), 
            )
            print(f"\nAssistant: {response}")
            print("\nYou: ", end='', flush=True)

    except KeyboardInterrupt:
        print("\nExiting...")
    except EOFError:
        print("\nInput stream closed. Exiting...")
    
    sys.exit(0)

if __name__ == "__main__":
    main()
