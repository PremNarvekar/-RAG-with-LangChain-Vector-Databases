import os

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

documents = [
    Document(
        page_content="Product SKU-7742X is our flagship router. It supports gigabit speeds and advanced QoS features.",
        metadata={"type": "product"},
    ),
    Document(
        page_content="For network connectivity issues, first check the ethernet cable and router status lights.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error code E_CONN_REFUSED indicates the server rejected the connection. Check firewall settings.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="The authentication process requires valid credentials. Use OAuth2 for secure API access.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Router configuration guide: Access the admin panel at 192.168.1.1 to modify settings.",
        metadata={"type": "config"},
    ),
    Document(
        page_content="WCAG 2.1 compliance requires all images to have alt text and sufficient color contrast.",
        metadata={"type": "compliance"},
    ),
]

print(f"Loaded {len(documents)} documents")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="hybrid_test",
)

vector_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

print("Vector retriever ready")

bm25_retriever = BM25Retriever.from_documents(
    documents,
    k=3,
)

print("BM25 retriever ready")


def hybrid_retriever(query, retrievers, weights, k=3, rrf_k=60):
    """Combine multiple retrievers using weighted Reciprocal Rank Fusion."""

    doc_scores = {}

    for retriever, weight in zip(retrievers, weights):
        results = retriever.invoke(query)

        for rank, doc in enumerate(results):
            key = doc.page_content
            rrf_score = weight * (1.0 / (rank + rrf_k))

            if key in doc_scores:
                doc_scores[key] = (
                    doc_scores[key][0] + rrf_score,
                    doc,
                )
            else:
                doc_scores[key] = (
                    rrf_score,
                    doc,
                )

    sorted_docs = sorted(
        doc_scores.values(),
        key=lambda x: x[0],
        reverse=True,
    )

    return [doc for _, doc in sorted_docs[:k]]


print("Hybrid retriever ready")