from typing import List, Dict

from retrieval_augmented_classification.knn_classifier import (
    KNNClassifier,
    DatasetVectorStore,
)
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def predict_single(item):
    rac, doc = item
    predicted_category = rac.predict(doc["text"])
    return predicted_category, doc["category"]


def calculate_accuracy(
    ground_truth_labels: List[str], predicted_labels: List[str]
) -> Dict[str, float]:
    """
    Calculates accuracy for single-label classification.

    Args:
        ground_truth_labels: A list of strings, where each string represents
                             the true class label for a document.
        predicted_labels: A list of strings, where each string represents
                          the predicted class label for a document.

    Returns:
        A dictionary containing the accuracy.
    """
    if len(ground_truth_labels) != len(predicted_labels):
        raise ValueError(
            "Ground truth and predicted labels lists must have the same length."
        )

    correct_predictions = 0
    total_predictions = len(ground_truth_labels)

    for true_label, predicted_label in zip(ground_truth_labels, predicted_labels):
        if true_label == predicted_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    return {"accuracy": accuracy}


# Example usage
if __name__ == "__main__":
    from retrieval_augmented_classification.load_dbpedia import (
        process_csv_to_documents,
    )
    from pathlib import Path

    data_path = Path(__file__).parents[1] / "data" / "datasets" / "dbpedia"
    dev_path = data_path / "DBPEDIA_val.csv"

    dev_documents = process_csv_to_documents(dev_path)

    # Initialize your classifier (replace with your actual classifier instance)
    # You would need to initialize the DatasetVectorStore first and pass it
    store = DatasetVectorStore()
    _classifier = KNNClassifier(store, k=5)

    _, subsample = train_test_split(
        dev_documents, test_size=500, shuffle=False, random_state=42
    )

    # Prepare data for multithreading
    prediction_tasks = [(_classifier, doc) for doc in subsample]

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
    # Accuracy: 0.8780
