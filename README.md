# retrieval_augmented_classification

This repository implements a retrieval-augmented classification (RAC) system. It combines the strengths of K-Nearest Neighbors (KNN) retrieval with the reasoning capabilities of a Large Language Model (LLM) to classify text documents.

## Overview

The core idea is to first retrieve relevant documents based on semantic similarity to the input text using a vector store (ChromaDB with BGE embeddings). Then, an LLM (specifically, a Gemini model via Langchain) leverages the retrieved context and the categories of the nearest neighbors to predict the most appropriate category for the input document.

This approach aims to improve classification accuracy by grounding the LLM's predictions in semantically similar examples from the dataset.

## Key Components

-   **`Makefile`**: Contains commands for common development tasks like installation, testing, formatting, and linting.
-   **`requirements-dev.txt`**: Lists development dependencies (e.g., `pytest`).
-   **`requirements.txt`**: Lists the core dependencies of the project (e.g., `langchain`, `google-genai`, `chromadb`).
-   **`retrieval_augmented_classification/__init__.py`**: An empty initialization file for the Python package.
-   **`retrieval_augmented_classification/clients.py`**: Initializes the Google Gemini embedding and LLM clients using API keys loaded from a `.env` file.
-   **`retrieval_augmented_classification/embedding.py`**: Implements the `GeminiAPIEmbeddings` class for generating text embeddings using the Google Gemini API.
-   **`retrieval_augmented_classification/evaluate_knn.py`**: Contains functions to evaluate the performance of the KNN classifier. It includes an example of loading the DBPedia validation set and calculating the accuracy of a `KNNClassifier`.
-   **`retrieval_augmented_classification/evaluate_rac.py`**: Contains functions to evaluate the performance of the `RAC` classifier. It loads the DBPedia validation set and uses multithreading to make predictions and calculate accuracy.
-   **`retrieval_augmented_classification/index_documents.py`**: A script to index the DBPedia training data into the ChromaDB vector store.
-   **`retrieval_augmented_classification/knn_classifier.py`**: Implements the `KNNClassifier` class, which uses a vector store to find the k-nearest neighbors of a document and predicts the category based on the majority vote of its neighbors' categories. It includes retry mechanisms for the vector store search.
-   **`retrieval_augmented_classification/load_dbpedia.py`**: Provides functions to load and process the DBPedia dataset from CSV files into a list of document dictionaries.
-   **`retrieval_augmented_classification/logger.py`**: Configures the `loguru` library for logging within the application, intercepting standard `logging` module calls.
-   **`retrieval_augmented_classification/retrieval_augmented_classifier.py`**: Implements the `RAC` class. This classifier retrieves relevant documents using a vector store and then uses an LLM with few-shot examples from the retrieved documents to predict the category of the input document. It leverages Pydantic for structured output from the LLM and includes retry mechanisms.
-   **`retrieval_augmented_classification/vector_store.py`**: Implements the `DatasetVectorStore` class, which uses ChromaDB with BGE embeddings to store and retrieve document embeddings. It provides methods for adding, deleting, searching, and wiping the document collection.

## Setup

### Prerequisites

-   Python 3.8 or higher
-   `conda` (recommended for managing dependencies)
-   A Google Cloud project with the Generative AI API enabled and an API key.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd retrieval_augmented_classification
    ```

2.  **Create a conda environment (optional but recommended):**
    ```bash
    conda create -n rac python=3.10  # Or your preferred Python version
    conda activate rac
    ```

3.  **Install PyTorch with CUDA (if you have a compatible GPU):**
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ```
    (Adjust the CUDA version based on your system.)

4.  **Install the remaining dependencies:**
    ```bash
    make install
    ```

5.  **Set up environment variables:**
    -   Create a `.env` file in the root directory of the repository.
    -   Add your Google API key to the `.env` file:
        ```
        GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
        ```
        Replace `YOUR_GOOGLE_API_KEY` with your actual API key.

## Data Download

https://www.kaggle.com/datasets/danofer/dbpedia-classes?select=DBP_wiki_data.csv 