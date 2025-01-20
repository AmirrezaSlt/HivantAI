from typing import Dict, Any
import os
from . import BaseDataSource

class TextFileDataSource:
    def __init__(self, base_directory: str):
        """Initialize with a base directory where text files are stored."""
        self.base_directory = base_directory

    def get_content(self, file_path: str) -> Dict[str, Any]:
        """
        Read content from a text file.
        
        Args:
            file_path: The full file path relative to base_directory
            
        Returns:
            Dict containing the file content and metadata
        """
        file_path = os.path.join(self.base_directory, file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            return {
                'content': content,
                'type': 'text/plain',
                'filename': os.path.basename(file_path),
                'size': os.path.getsize(file_path)
            }
        except FileNotFoundError:
            raise ValueError(f"Text file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading text file: {str(e)}")

    def get_resource_link(self, uri: str) -> str:
        """
        Get the full file path for a text resource.
        
        Args:
            uri: The filename or path relative to base_directory
            
        Returns:
            Full path to the text file
        """
        return os.path.join(self.base_directory, uri)