# Local RAG pipeline project for a dataset about sales

This is a University of Helsinki course project concerning data warehousing and business intelligence. It's a local RAG pipeline that analyzes sales data, namely the [Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) from Kaggle.

It can be asked business intelligence questions about the data in natural language to get answers from a local LLM, specifically Llama 3.2 3B via Ollama.

You only need Python, Ollama and 4 GB of RAM to run this.

Note that the model is fairly small and thus hallucinates more. It also has no memory of past queries.

## Setup

### 1. Clone the project and download the dataset

After cloning this project, you will need a Kaggle account to download the dataset: [https://www.kaggle.com/datasets/vivek468/superstore-dataset-final](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

Place that CSV file inside this project's data folder.

### 2. Install Ollama

You will need to install Ollama on your system and then pull Llama 3.2 3B.

```
ollama pull llama3.2:3b
```

### 3. Install dependencies

Inside the project folder, activate venv and install the dependencies in requirements.txt.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Chunk and embed the data

First run chunk.py which produces chunks.json from the CSV. Then run embed.py which uses that to produce chroma_db.

```
python chunk.py
python embed.py
```

Both steps only need to be run once. After this you can run the proper RAG.

## Run

In a second terminal, separate from the venv one, run ollama:
```
ollama serve
```

Then in the original venv terminal run either rag.py or queries.py (latter just runs rag.py but with many already made queries).

```
python rag.py
```
or
```
python queries.py
```
Edit the files to change the queries. There are three things to modify in a query:
1. Question itself. This isn't very complex and is just a regular question in regular language about the dataset.
2. The top-k retrieval value which is n_results in rag.py and queries.py. This is how many chunks you're looking at.
3. Metadata filter type. This is the specific kind of chunks you're looking at. There are five types: transactions, monthly, categories, regions and rankings.

### Sample output

```
Q: Which product category is the most profitable and which is the least?
Based on the provided data, here are the answers to your question:

1. The most profitable product category is Technology with a profit of $145,454.95.
2. The least profitable product category is Furniture with a profit of $18,451.27.
```
