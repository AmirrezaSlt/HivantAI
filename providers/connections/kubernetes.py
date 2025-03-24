from typing import Optional, Dict, Any
from server import client, config
from agent.connections import BaseConnection

class KubernetesConnection(BaseConnection):
    _instances: Dict[str, 'KubernetesConnection'] = {}
    
    def __init__(self, kube_config: Optional[Dict[str, Any]] = None, namespace: Optional[str] = None, context: Optional[str] = None):
        super().__init__()
        self.kube_config = kube_config
        self.namespace = namespace
        self.context = context
        self._api_client = None
        
    @property
    def api_client(self):
        if self._api_client is None:
            self.load_config()
            self._api_client = client.ApiClient()
        return self._api_client

    @classmethod
    def get_connection(cls, kube_config: Optional[Dict[str, Any]] = None, namespace: Optional[str] = None, context: Optional[str] = None) -> 'KubernetesConnection':
        # Create a unique key for this connection configuration
        key = f"{hash(str(kube_config))}-{namespace}-{context}"
        if key not in cls._instances:
            cls._instances[key] = cls(kube_config=kube_config, namespace=namespace, context=context)
        return cls._instances[key]

    def load_config(self):
        try:
            if self.kube_config:
                config_obj = client.Configuration.from_dict(self.kube_config)
            else:
                config.load_kube_config(context=self.context)
                config_obj = client.Configuration()
                config.load_kube_config(client_configuration=config_obj, context=self.context)

            if self.namespace:
                config_obj.api_key['namespace'] = self.namespace

            client.Configuration.set_default(config_obj)
        except config.config_exception.ConfigException as e:
            raise ValueError(f"Error loading Kubernetes config: {e}")

    def close(self):
        if self._api_client:
            self._api_client.close()
            self._api_client = None

    def __del__(self):
        self.close()
