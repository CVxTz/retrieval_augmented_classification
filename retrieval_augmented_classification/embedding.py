import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List

from langchain_core.embeddings import Embeddings

from retrieval_augmented_classification.clients import embedding_client


class GeminiAPIEmbeddings(Embeddings):
    def __init__(self, model: str = "text-embedding-004"):
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        with ThreadPoolExecutor(max_workers=20) as executor:
            return list(executor.map(self.embed_query, texts))

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        response = embedding_client.models.embed_content(
            model=self.model,
            contents=str(text)[:10000] if text else uuid.uuid4().hex,
        )
        return response.embeddings[0].values


if __name__ == "__main__":
    embeddings = GeminiAPIEmbeddings()

    embed = embeddings.embed_query("sssss U")

    print(embed)

    for x in embed:
        print(x)

    embeds = embeddings.embed_documents(["sssss U"])

    print(embeds)

    for x in embeds:
        print(x)
