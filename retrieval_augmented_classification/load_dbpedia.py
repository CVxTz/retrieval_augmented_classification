from pathlib import Path
from typing import Union
import csv


def process_csv_to_documents(csv_file_path: Union[str, Path]) -> list[dict]:
    """
    Processes a CSV file to create a list of document dictionaries.
    Each document dictionary will have 'text' and 'category' keys.
    The 'text' value comes from the 'text' column in the CSV,
    and the 'category' value comes from the 'l3' column.

    Args:
        csv_file_path: Path to the CSV file.

    Returns:
        A list of dictionaries, where each dictionary represents a document.
    """

    documents = []
    with open(csv_file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)  # Use DictReader for column access by name
        for row in reader:
            document = {"text": row["text"], "category": row["l3"]}
            documents.append(document)
    return documents


def get_all_categories(csv_file_path: Union[str, Path]) -> set:
    """
    Processes a CSV file to create a list of document dictionaries.
    Each document dictionary will have 'text' and 'category' keys.
    The 'text' value comes from the 'text' column in the CSV,
    and the 'category' value comes from the 'l3' column.

    Args:
        csv_file_path: Path to the CSV file.

    Returns:
        A list of dictionaries, where each dictionary represents a document.
    """

    categories = set()
    with open(csv_file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)  # Use DictReader for column access by name
        for row in reader:
            categories.add(row["l3"])
    return categories


if __name__ == "__main__":
    # Example usage:
    csv_file = "example.csv"  # Replace with your CSV file path

    # Create a dummy CSV file for testing
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "l1", "l2", "l3"])
        writer.writerow(["This is document 1 about cats.", "Animal", "Mammal", "Cat"])
        writer.writerow(["This is document 2 about dogs.", "Animal", "Mammal", "Dog"])
        writer.writerow(["This is document 3 about birds.", "Animal", "Bird", "Bird"])

    _documents = process_csv_to_documents(csv_file)
    print("All Documents:")
    for doc in _documents:
        print(doc)
