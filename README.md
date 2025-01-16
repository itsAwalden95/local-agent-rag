## Local Multi Agent RAG

A Multi-Agent RAG (Retrieval-Augmented Generation) Research Assistant powered by LangGraph, designed to tackle complex research queries with precision. The system employs intelligent agents that break down research tasks into strategic steps, leverage specialized tools, and implement rigorous fact-checking mechanisms to ensure reliable, accurate responses.

## Getting Started

To get started with this project, follow these steps:

First, clone the repository to your local machine:

```bash
git clone https://github.com/itsAwalden95/local-agent-rag.git
cd local-agent-rag
```

```bash
pip install -r requirements.txt
```

Then open the config.yml file located in the root directory of the project. Set the value of load_documents to **true** to ensure the necessary documents are loaded into the vector database:

Then run:

```bash
python3 -m retriever.retriever
```

Once the PDF has been processed and indexed, you can start the application by running the following command:

```bash
python3 app.py
```

Now ask your question based on the document: https://sustainability.google/reports/google-2024-environmental-report/
