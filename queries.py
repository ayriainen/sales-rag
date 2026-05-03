"""
This has queries for the rag that you can just run via "python queries.py" after chunk and embed.
You can add to the list or remove from the list.
There are three things you can change: the question, the n_results (top-k) and the filter type.
Filter types are transactions, monthly, categories, regions and rankings.
"""
from rag import ask

queries = [
    ("How do the four regions compare in terms of profit margin?",
     4, "regions"),
    ("What subcategories have the highest profit margins?",
     17, "categories"),
    ("How has overall sales and profit grown year over year?",
     3, "rankings"),
    ("Which subcategories have both a high average discount and a negative or low profit margin?",
     17, "categories"),
    ("Which product category is the most profitable and which is the least?",
     3, "rankings"),
    ("Which month had the highest profit margin in 2017?",
     48, "monthly"),
]

for question, n_results, filter_type in queries:
    print()
    print(f"Q: {question}")
    answer, _ = ask(question, n_results=n_results, filter_type=filter_type)
    print(answer)
