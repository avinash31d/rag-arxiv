import arxiv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
import ollama
from uuid import uuid4
import gradio as gr
from dotenv import load_dotenv
import os
load_dotenv()

# Constants
EMBEDDING_MODEL="nomic-embed-text"
VECTOR_DB_COLLECTION = "arxiv_papers"
VECTOR_DB_API_KEY =  os.getenv("VECTOR_DB_API_KEY")

# Initialize Qdrant
qdrant_client = QdrantClient(
    url=  os.getenv("VECTOR_DB_API_URL"),
    api_key=VECTOR_DB_API_KEY,
)

# Check if the collection exists, if not, create it
if not qdrant_client.collection_exists(VECTOR_DB_COLLECTION):
    qdrant_client.create_collection(
        collection_name=VECTOR_DB_COLLECTION,
        vectors_config=qmodels.VectorParams(size=768, distance=qmodels.Distance.COSINE),
    )

def fetch_arxiv_papers(query, max_results=10):
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    papers = []
    for result in search.results():
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "link": result.entry_id
        })
    return papers


def embed_text_with_ollama(text):
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    if "embedding" not in response:
        raise ValueError("Failed to generate embedding from Ollama API")
    return response["embedding"]


def index_paper_in_qdrant(paper, embedding):
    qdrant_client.upsert(
        collection_name=VECTOR_DB_COLLECTION,
        points=[
            qmodels.PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={"title": paper["title"], "abstract": paper["abstract"], "link": paper["link"]},
            )
        ],
    )


def retrieve_similar_papers(query, top_k=5):
    query_embedding = embed_text_with_ollama(query)
    search_result = qdrant_client.search(
        collection_name=VECTOR_DB_COLLECTION,
        query_vector=query_embedding,
        limit=top_k,
    )
    return search_result

def respond_to_query(user_query):
    # Fetch papers based on user query
    papers = fetch_arxiv_papers(user_query)

    # Index papers into Qdrant
    for paper in papers:
        embedding = embed_text_with_ollama(paper["abstract"])
        index_paper_in_qdrant(paper, embedding)

    # Retrieve similar papers based on user query
    similar_papers = retrieve_similar_papers(user_query)

    # Format the response
    response = []
    for result in similar_papers:
        title = result.payload["title"]
        link = result.payload["link"]
        abstract = result.payload["abstract"]

        response.append(f"Title: {title}\nLink: {link}\nAbstract: {abstract}\n\n---\n")

    return "\n".join(response)


# Gradio Interface
def main():
    # Gradio interface for taking user query and displaying results
    interface = gr.Interface(
        fn=respond_to_query,
        inputs=gr.Textbox(label="Enter your query", placeholder="Search papers on arXiv"),
        outputs=gr.Textbox(label="Similar Papers"),
        live=True,
    )

    interface.launch()

if __name__ == "__main__":
    main()