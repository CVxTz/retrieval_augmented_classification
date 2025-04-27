from typing import List
from uuid import uuid4

from langchain_core.documents import Document
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import torch
from tqdm import tqdm
from chromadb.config import Settings
from retrieval_augmented_classification.logger import logger


class DatasetVectorStore:
    """ChromaDB vector store for PublicationModel objects with SentenceTransformers embeddings."""

    def __init__(
        self,
        db_name: str = "retrieval_augmented_classification",  # Using db_name as collection name in Chroma
        collection_name: str = "classification_dataset",
        persist_directory: str = "chroma_db",  # Directory to persist ChromaDB
    ):
        self.db_name = db_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Determine if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={
                "device": device,
                "batch_size": 100,
            },  # Adjust batch_size as needed
        )

        # Initialize Chroma vector store
        self.client = PersistentClient(
            path=self.persist_directory, settings=Settings(anonymized_telemetry=False)
        )
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, documents: List) -> None:
        """
        Add multiple documents to the vector store.

        Args:
            documents: List of dictionaries containing document data.  Each dict needs a "text" key.
        """

        local_documents = []
        ids = []

        for doc_data in documents:
            if not doc_data.get("id"):
                doc_data["id"] = str(uuid4())

            local_documents.append(
                Document(
                    page_content=doc_data["text"],
                    metadata={k: v for k, v in doc_data.items() if k != "text"},
                )
            )
            ids.append(doc_data["id"])

        batch_size = 100  # Adjust batch size as needed
        for i in tqdm(range(0, len(documents), batch_size)):
            batch_docs = local_documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            # Chroma's add_documents doesn't directly support pre-defined IDs. Upsert instead.
            self._upsert_batch(batch_docs, batch_ids)

    def _upsert_batch(self, batch_docs: List[Document], batch_ids: List[str]):
        """Upsert a batch of documents into Chroma.  If the ID exists, it updates; otherwise, it creates."""
        texts = [doc.page_content for doc in batch_docs]
        metadatas = [doc.metadata for doc in batch_docs]

        self.vector_store.add_texts(texts=texts, metadatas=metadatas, ids=batch_ids)

    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        self.vector_store.delete(ids=[document_id])
        return True

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search documents by semantic similarity."""
        results = self.vector_store.similarity_search(query, k=k)
        return results

    def wipe_collection(self):
        """Deletes the entire Chroma collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Successfully wiped collection: {self.collection_name}")
        except ValueError as e:
            logger.warning(
                f"Collection {self.collection_name} not found or error during deletion: {e}"
            )

        # Re-initialize the collection
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )


# Example Usage (Illustrative)
if __name__ == "__main__":
    # Create some dummy documents
    _documents = [
        {"id": "1", "text": "This is document 1 about cats.", "category": "pets"},
        {"id": "2", "text": "This is document 2 about dogs.", "category": "pets"},
        {"id": "3", "text": "This is document 3 about birds.", "category": "animals"},
    ]

    # Initialize the vector store
    vector_store = DatasetVectorStore(
        db_name="my_chroma_db",
        collection_name="my_collection",
        persist_directory="my_chroma_db_directory",
    )

    # Add the documents to the vector store
    vector_store.add_documents(_documents)

    # Search for documents related to "pets"
    _results = vector_store.search("pets", k=2)
    print("Search results:", _results)

    # Wipe the collection
    vector_store.wipe_collection()
    print("Collection wiped.")

    # Search again after wiping (should be empty)
    _results = vector_store.search("pets", k=2)
    print("Search results after wiping:", _results)

    # Add new documents after wiping
    new_documents = [
        {"id": "4", "text": "New document about elephants.", "category": "animals"}
    ]
    vector_store.add_documents(new_documents)

    # Search again to confirm new documents are present
    results = vector_store.search("elephants", k=2)
    print("Search results after adding new documents:", results)
