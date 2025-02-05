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
from agent.toolkit.code_execution import CodeExecutionTool
import json 
def main():

    kubernetes_conn = KubernetesConnection.get_connection()

    cognitive_engine = CognitiveEngine(
        SYSTEM_PROMPT="""
        You are an AI assistant that tries to help the user with their Kubernetes problems. 
        You already have access to the main kubernetes cluster (dg-cluster) by default so assume that's the cluster you're working with and prefer to get data yourself than asking the user for it if possible. 
        You can use the code execution tool to run python code that uses the kubernetes client to run kubernetes commands and get the output.
        Try to do incremental steps and get to a good response and feel free to use the tools multiple times, the previous steps taken will be provided to you. 
        Keep your codes small and atomic and try to debug through multiple steps rather than one large block of code.
        """,
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
        # REFERENCE_DOCUMENTS=[
        #     TextFileReferenceDocument(id="kubectl_debug_guide", file_path="providers/data/kubectl_debug.txt"),
        #     TextFileReferenceDocument(id="contact", file_path="providers/data/contact.txt")
        # ]
    )
    
    toolkit = Toolkit(
        TOOLS=[
            # KubernetesPodTool(
                # id="kubernetes_pod",
                # kubernetes_connection=kubernetes_conn
            # ),
            # KubernetesLogsTool(
                # id="kubernetes_logs",
                # kubernetes_connection=kubernetes_conn
            # ),
            CodeExecutionTool(
                id="code_execution",
                server_address="http://executor:8000",
                description="""
                Runs Python code and returns the output or error message, has the following libraries installed: kubernetes. 
                Do NOT include explanations or comments in your code.
                You cannot handle large outputs, so if you're retrieving logs or reading files, etc, you should limit the output.
                You can only see the print statements that you have in your code so use them whenever you need to get a response.
                """
            )
        ]
    )

    state = None
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
        with open("providers/data/kubectl_guide.txt", "rb") as f:
            kubectl_guide_content = f.read()
        
        for line in sys.stdin:
            user_input = line.strip()
            
            if user_input.lower() in ['quit', 'exit']:
                break
            elif not user_input:
                print("\nYou: ", end='', flush=True)
                continue

            attachments = {
                "kubectl_guide.txt": kubectl_guide_content
            }
            response, state = agent.respond(
                input=Input(message=user_input, attachments=attachments), 
                state=state
            )
            with open("state.json", "w") as f:
                json_list = "\n".join([str(step.model_dump_json()) for step in state.trail])
                f.write(json_list)
            print(f"\nAssistant: {response}")
            print("\nYou: ", end='', flush=True)

    except KeyboardInterrupt:
        print("\nExiting...")
    except EOFError:
        print("\nInput stream closed. Exiting...")
    
    sys.exit(0)

if __name__ == "__main__":
    main()
