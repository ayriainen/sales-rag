"""
This is the main rag part. It can be called directly via "python rag.py" or via "python queries.py".
If you use it directly, edit the values at the bottom, otherwise edit queries.py.
Run prep.py and then setup.py before this, although they only need to be run once.
"""
import chromadb
from chromadb.utils import embedding_functions
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

client = chromadb.PersistentClient(path="./chroma_db")
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = client.get_collection("sales_data", embedding_function=emb_fn)

# temperature is low to get only precise, comparable answers for a project like this
# 0.1 temp seems to be conventionally preferred over 0.0 but is said to be nigh same
# num_predict limits response token size as in word size
llm = OllamaLLM(model="llama3.2:3b", temperature=0.1, num_predict=512)

prompt_temp = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use only the sales data below to answer the question.
Do not use outside knowledge. Be specific with numbers from the data.
If the data does not support a clear answer, say so briefly.
Data:
{context}
Question:
{question}
Answer:""",
)

def ask(question, n_results=6, filter_type=None):
    """Main RAG function, n_results is top-k and filter_type is metadata filter"""
    where = {"type": filter_type} if filter_type else None
    results = collection.query(query_texts=[question], n_results=n_results, where=where)
    context = "\n\n".join(results["documents"][0])
    prompt = prompt_temp.format(context=context, question=question)
    response = llm.invoke(prompt)
    return response, context

if __name__ == "__main__":
    # edit this question to your liking
    Q = "Which region is most profitable?"
    # edit n_results to your preferred chunk size and filter_type to the filter fitting the question
    answer, _ = ask(Q, n_results=6, filter_type="regions")
    print(answer)
