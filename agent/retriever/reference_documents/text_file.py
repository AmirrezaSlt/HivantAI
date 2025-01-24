from typing import Dict, Any
import os
from . import BaseReferenceDocument

class TextFileReferenceDocument(BaseReferenceDocument):
    def __init__(self, id: str, file_path: str):
        """Initialize with a resource ID and file path."""
        super().__init__(id)
        self._file_path = os.path.abspath(file_path)
        if not os.path.exists(self._file_path):
            raise ValueError(f"Text file not found: {self._file_path}")

    @property
    def link(self) -> str:
        """Return the absolute file path as the resource identifier"""
        return self._file_path

    @property
    def data(self) -> str:
        """Read and return the file content"""
        try:
            with open(self._file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except Exception as e:
            raise ValueError(f"Error reading text file: {str(e)}")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return file metadata"""
        return {
            'id': self.id,
            'filename': os.path.basename(self._file_path),
            'size': os.path.getsize(self._file_path),
            'type': 'text/plain'
        }
