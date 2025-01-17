from typing import Type, Dict, Any, Optional
from pydantic import BaseModel, Field
from kubernetes import client
from agent.tools import BaseTool
from ..connections.kubernetes import KubernetesConnection

class KubernetesLogsInputs(BaseModel):
    name: str = Field(..., description="The name of the pod")
    namespace: str = Field(..., description="The namespace of the pod")
    container: Optional[str] = Field(None, description="The container name (if pod has multiple containers)")
    tail_lines: Optional[int] = Field(None, description="Number of lines to return from the end of the logs")
    previous: bool = Field(False, description="Return previous terminated container logs")

class KubernetesLogsTool(BaseTool):
    def __init__(self, kubernetes_connection: KubernetesConnection):
        self.connection: KubernetesConnection = kubernetes_connection
        self.core_v1_api: client.CoreV1Api = client.CoreV1Api(self.connection.api_client)

    @property
    def input_model(self) -> Type[BaseModel]:
        return KubernetesLogsInputs

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "string",
            "description": "Pod logs as a string"
        }

    @property
    def description(self) -> str:
        return "Retrieve logs from a Kubernetes pod, with options for specific containers, tail lines, and previous container logs."

    def _invoke(self, inputs: KubernetesLogsInputs) -> str:
        try:
            logs = self.core_v1_api.read_namespaced_pod_log(
                name=inputs.name,
                namespace=inputs.namespace,
                container=inputs.container,
                tail_lines=inputs.tail_lines,
                previous=inputs.previous
            )
            return logs
        except client.ApiException as e:
            raise ValueError(f"Error fetching pod logs: {e}")