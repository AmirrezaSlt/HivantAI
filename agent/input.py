from typing import Dict, Any
from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

class InputType(Enum):
    TEXT = "text"
    # Extensible for future types:
    # IMAGE = "image"
    # AUDIO = "audio"
    # etc.

class Input(BaseModel):
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)
    attachments: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    @field_validator('attachments', mode='before')
    def process_attachments(cls, value: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Process raw binary attachments into structured data.
        
        Args:
            value: Dictionary of filename to binary content mappings
        Returns:
            Processed attachments dictionary
        """
        processed = {}
        for filename, binary_content in value.items():
            if isinstance(binary_content, bytes):  # Only process if raw binary content
                if filename.lower().endswith(('.txt', '.py', '.md', '.json')):
                    input_type = InputType.TEXT
                    text_content = binary_content.decode('utf-8')
                    
                    processed[filename] = {
                        'type': input_type.value,
                        'content': text_content,
                    }
                else:
                    raise NotImplementedError(f"File type for {filename} not supported")
            else:
                processed[filename] = binary_content  # Keep already processed attachments as-is
                
        return processed
