from retrieval_augmented_classification.load_dbpedia import process_csv_to_documents
from retrieval_augmented_classification.vector_store import DatasetVectorStore
from pathlib import Path

if __name__ == "__main__":
    store = DatasetVectorStore()

    data_path = Path(__file__).parents[1] / "data" / "datasets" / "dbpedia"
    train_path = data_path / "DBPEDIA_train.csv"

    processed_csv = process_csv_to_documents(train_path)

    store.wipe_collection()

    store.add_documents(processed_csv)

    # Search
    results = store.search("Mountain")
    for doc in results:
        print(doc.page_content)
