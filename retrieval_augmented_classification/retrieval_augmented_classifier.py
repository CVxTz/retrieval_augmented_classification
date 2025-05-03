from typing import Optional
from pydantic import BaseModel, Field
from collections import Counter

from retrieval_augmented_classification.vector_store import DatasetVectorStore
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class PredictedCategories(BaseModel):
    """
    Pydantic model for the predicted categories from the LLM.
    """

    reasoning: str = Field(description="Explain your reasoning")
    predicted_category: str = Field(description="Category")


class RAC:
    """
    A hybrid classifier combining K-Nearest Neighbors retrieval with an LLM for multi-class prediction.
    Finds top K neighbors, uses top few-shot for context, and uses all neighbor categories
    as potential prediction candidates for the LLM.
    """

    def __init__(
        self,
        vector_store: DatasetVectorStore,
        llm_client,
        knn_k_search: int = 30,
        knn_k_few_shot: int = 5,
    ):
        """
        Initializes the classifier.

        Args:
            vector_store: An instance of DatasetVectorStore with a search method.
            llm_client: An instance of the LLM client capable of structured output.
            knn_k_search: The number of nearest neighbors to retrieve from the vector store.
            knn_k_few_shot: The number of top neighbors to use as few-shot examples for the LLM.
                           Must be less than or equal to knn_k_search.
        """

        self.vector_store = vector_store
        self.llm_client = llm_client
        self.knn_k_search = knn_k_search
        self.knn_k_few_shot = knn_k_few_shot

    @retry(
        stop=stop_after_attempt(3),  # Retry LLM call a few times
        wait=wait_exponential(multiplier=1, min=2, max=5),  # Shorter waits for demo
    )
    def predict(self, document_text: str) -> Optional[str]:
        """
        Predicts the relevant categories for a given document text using KNN retrieval and an LLM.

        Args:
            document_text: The text content of the document to classify.

        Returns:
            A list of predicted categories from the LLM based on the KNN context.
            Returns an empty list if no neighbors are found or LLM prediction fails.
        """
        neighbors = self.vector_store.search(document_text, k=self.knn_k_search)

        all_neighbor_categories = set()
        valid_neighbors = []  # Store neighbors that have metadata and categories
        for neighbor in neighbors:
            if (
                hasattr(neighbor, "metadata")
                and isinstance(neighbor.metadata, dict)
                and "category" in neighbor.metadata
            ):
                all_neighbor_categories.add(neighbor.metadata["category"])
                valid_neighbors.append(neighbor)
            else:
                pass  # Suppress warnings for cleaner demo output

        if not valid_neighbors:
            return None

        category_counts = Counter(all_neighbor_categories)
        ranked_categories = [
            category for category, count in category_counts.most_common()
        ]

        if not ranked_categories:
            return None

        few_shot_neighbors = valid_neighbors[: self.knn_k_few_shot]

        messages = []

        system_prompt = f"""You are an expert multi-class classifier. Your task is to analyze the provided document text and assign the most relevant category from the list of allowed categories.
You MUST only return categories that are present in the following list: {ranked_categories}.
If none of the allowed categories are relevant, return an empty list.
Return the categories by likelihood (more confident to least confident).
Output your prediction as a JSON object matching the Pydantic schema: {PredictedCategories.model_json_schema()}.
"""
        messages.append(SystemMessage(content=system_prompt))

        for i, neighbor in enumerate(few_shot_neighbors):
            messages.append(
                HumanMessage(content=f"Document: {neighbor.page_content}")
            )
            expected_output_json = PredictedCategories(
                reasoning="Your reasoning here",
                predicted_category=neighbor.metadata["category"]
            ).model_dump_json()
            # Simulate the structure often used with tool calling/structured output

            ai_message_with_tool = AIMessage(
                content=expected_output_json,
            )

            messages.append(ai_message_with_tool)

        # Final user message: The document text to classify
        messages.append(HumanMessage(content=f"Document: {document_text}"))

        # Configure the client for structured output with the Pydantic schema
        structured_client = self.llm_client.with_structured_output(PredictedCategories)
        llm_response: PredictedCategories = structured_client.invoke(messages)

        predicted_category = llm_response.predicted_category

        return predicted_category if predicted_category in ranked_categories else None
