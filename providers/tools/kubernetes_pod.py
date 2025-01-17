from typing import Type, Dict, Any
from pydantic import BaseModel, Field
from kubernetes import client
from agent.tools import BaseTool
from ..connections.kubernetes import KubernetesConnection

class KubernetesPodInputs(BaseModel):
    name: str = Field(..., description="The name of the pod")
    namespace: str = Field(..., description="The namespace of the pod")

class KubernetesPodTool(BaseTool):
    def __init__(self, kubernetes_connection: KubernetesConnection):
        self.connection: KubernetesConnection = kubernetes_connection
        self.core_v1_api: client.CoreV1Api = client.CoreV1Api(self.connection.api_client)

    @property
    def input_model(self) -> Type[BaseModel]:
        return KubernetesPodInputs

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "description": "kubernetes.client.V1Pod object"
        }

    @property
    def description(self) -> str:
        return "Look up information about a Kubernetes pod."

    def _invoke(self, inputs: KubernetesPodInputs) -> Dict[str, Any]:
        try:
            pod = self.core_v1_api.read_namespaced_pod(name=inputs.name, namespace=inputs.namespace)
            return pod
        except client.ApiException as e:
            raise ValueError(f"Error fetching pod information: {e}")
