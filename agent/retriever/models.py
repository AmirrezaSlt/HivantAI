from pydantic import BaseModel

class Document(BaseModel):
    data_source: str
    content: str
