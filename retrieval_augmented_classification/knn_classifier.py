import functools
from collections import Counter
from typing import Optional
from retrieval_augmented_classification.vector_store import DatasetVectorStore
from tenacity import retry, stop_after_attempt, wait_exponential


class KNNClassifier:
    """
    A K-Nearest Neighbors multi-label classifier using a vector store.
    Predicts tags based on the frequency of tags among the k nearest neighbors.
    """

    def __init__(self, vector_store: DatasetVectorStore, k: int = 5):
        """
        Initializes the classifier with a vector store and the number of neighbors.

        Args:
            vector_store: An instance of DatasetVectorStore with a search method.
            k: The number of nearest neighbors to consider for classification.
        """
        if not isinstance(vector_store, DatasetVectorStore):
            raise TypeError("vector_store must be an instance of DatasetVectorStore")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")

        self.vector_store = vector_store
        self.k = k

    @functools.lru_cache(maxsize=None)
    @retry(
        stop=stop_after_attempt(10),  # Stop after 5 attempts
        wait=wait_exponential(
            multiplier=1, min=4, max=10
        ),  # Wait 2^n * 1 seconds between retries, maxing at 10s
        # You can add 'retry' parameter here to specify exceptions to retry on,
        # e.g., retry=retry_if_exception_type(SomeNetworkError)
    )
    def predict(self, document_text: str) -> Optional[str]:
        """
        Predicts the relevant tags for a given document text.

        This method is decorated with tenacity's retry mechanism with exponential backoff
        to handle potential transient errors during the vector store search operation.

        Args:
            document_text: The text content of the document to classify.

        Returns:
            A list of predicted tags, ranked by frequency in descending order.
        """
        # Search for the k nearest neighbors
        neighbors = self.vector_store.search(document_text, k=self.k)

        if not neighbors:
            print("No neighbors found.")
            return None

        # Collect all tags from the neighbors' metadata
        all_categories = []
        for neighbor in neighbors:
            if hasattr(neighbor, "metadata") and "category" in neighbor.metadata:
                all_categories.append(neighbor.metadata["category"])
            else:
                print(f"Warning: Neighbor missing metadata or tags: {neighbor}")

        if not all_categories:
            print("No tags found in neighbors' metadata.")
            return None

        # Count the frequency of each tag
        category_counts = Counter(all_categories)

        # Rank tags by frequency and return
        # sorted(key=lambda item: item[1], reverse=True) sorts by count
        # [tag for tag, count in ...] extracts just the tag names
        ranked_tags = [tag for tag, count in category_counts.most_common()]

        return ranked_tags[0]


# Example usage (assuming you have DatasetVectorStore properly implemented and loaded)
if __name__ == "__main__":
    # Initialize your actual vector store here
    # store = DatasetVectorStore() # Use your actual store initialization

    # Using the placeholder store for demonstration
    store = DatasetVectorStore()

    # Initialize the classifier
    classifier = KNNClassifier(store, k=5)  # Use k=3 for the dummy data example

    # Document text to classify
    document_to_classify = "Mount everest"

    # Get predictions
    category = classifier.predict(document_to_classify)

    print(f"\nDocument to classify: '{document_to_classify}'")
    print(f"Predicted category (majority vote): {category}")

    # Example with a different query
    document_to_classify_2 = "Paris, France"
    category_2 = classifier.predict(document_to_classify_2)
    print(f"\nDocument to classify: '{document_to_classify_2}'")
    print(f"Predicted category (majority vote): {category_2}")
