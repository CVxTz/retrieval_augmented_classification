from retrieval_augmented_classification.retrieval_augmented_classifier import RAC
from retrieval_augmented_classification.vector_store import DatasetVectorStore

from retrieval_augmented_classification.evaluate_knn import calculate_accuracy
from retrieval_augmented_classification.clients import llm_client
from retrieval_augmented_classification.load_dbpedia import (
    process_csv_to_documents,
)
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def predict_single(item):
    rac, doc = item
    predicted_category = rac.predict(doc["text"])
    return predicted_category, doc["category"]


if __name__ == "__main__":
    from pathlib import Path

    data_path = Path(__file__).parents[1] / "data" / "datasets" / "dbpedia"
    dev_path = data_path / "DBPEDIA_val.csv"

    dev_documents = process_csv_to_documents(dev_path)

    store = DatasetVectorStore()

    _rac = RAC(
        vector_store=store,
        llm_client=llm_client,
        knn_k_search=50,
        knn_k_few_shot=10,
    )
    print(
        f"Initialized rac with knn_k_search={_rac.knn_k_search}, knn_k_few_shot={_rac.knn_k_few_shot}."
    )

    _, subsample = train_test_split(
        dev_documents, test_size=500, shuffle=False, random_state=42
    )

    # Prepare data for multithreading
    prediction_tasks = [(_rac, doc) for doc in subsample]

    _predicted_labels = []
    _ground_truth_labels = []

    print("Making predictions for dev documents using ThreadPoolExecutor...")
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(predict_single, task) for task in prediction_tasks]
        for future in tqdm(futures, total=len(prediction_tasks)):
            predicted, ground_truth = future.result()
            _predicted_labels.append(predicted)
            _ground_truth_labels.append(ground_truth)

    # Calculate and print accuracy
    evaluation_results = calculate_accuracy(_ground_truth_labels, _predicted_labels)

    print("\nEvaluation Results:")
    print(f"Accuracy: {evaluation_results['accuracy']:.4f}")

    # Evaluation
    # Results:
    # Accuracy: 0.9640
    # 500 samples in 15 seconds => 30ms per sample throughput
    # latency around 1s
