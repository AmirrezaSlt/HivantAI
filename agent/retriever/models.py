from typing import List
from pydantic import BaseModel

class RelevantDocument(BaseModel):
    retriever_id: str
    uri: str
    link: str
    score: float
    content: str

class RelevantDocuments(BaseModel):
    documents: List[RelevantDocument]
    
    def add_document(self, document: RelevantDocument) -> None:
        """Add a single RelevantDocument to the documents list.
        
        Args:
            document (RelevantDocument): The document to add to the collection
        """
        self.documents.append(document)
