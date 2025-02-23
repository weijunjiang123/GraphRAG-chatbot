# GraphRAG Project

This project implements a Retrieval-Augmented Generation (RAG) service using various machine learning models and vector stores. The service is designed to handle queries and provide responses based on the indexed documents.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/weijunjiang123/graphRAG.git
    cd graphRAG
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Start the backend service:
    ```sh
    python src/RAG/main.py
    ```

2. Start the frontend service:
    ```sh
    python streamlit run src/RAG/frontend.py
    ```

## Configuration

Configuration settings are managed using environment variables.
You can set these variables in a `./src/RAG/core/config.py` file or 
directly in your environment.

