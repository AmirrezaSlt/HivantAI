import os
from typing import Optional, Dict
from agent.connections import BaseConnection
from requests import Session

class NotionConnection(BaseConnection):
    _instances: Dict[str, 'NotionConnection'] = {}
    
    def __init__(self, workspace: Optional[str] = None, api_key: Optional[str] = None):
        super().__init__()
        self.workspace = workspace or os.environ.get("NOTION_WORKSPACE")
        if not self.workspace:
            raise ValueError("Workspace must be provided or set in NOTION_WORKSPACE environment variable")
        self.api_key = api_key or os.environ["NOTION_API_KEY"]
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": "2022-06-28",
            "Content-Type": "application/json"
        }
        self._session = None

    @property
    def session(self) -> Session:
        if self._session is None:
            self._session = Session()
            self._session.headers.update(self.headers)
        return self._session

    @classmethod
    def get_connection(cls, workspace: Optional[str] = None, api_key: Optional[str] = None) -> 'NotionConnection':
        # Create a unique key for this connection configuration
        key = f"{workspace}-{api_key}"
        if key not in cls._instances:
            cls._instances[key] = cls(workspace=workspace, api_key=api_key)
        return cls._instances[key]

    def close(self):
        if self._session:
            self._session.close()
            self._session = None

    def __del__(self):
        self.close()