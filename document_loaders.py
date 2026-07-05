import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    DirectoryLoader,
    PyPDFLoader,
)

load_dotenv()


# def load_text_file():
#     with tempfile.NamedTemporaryFile(
#         delete=False, suffix=".txt", mode="w", encoding="utf-8"
#     ) as temp_file:
#         temp_file.write(
#             "Hello, this is a sample text file.\n"
#             "This file is used to demonstrate the TextLoader."
#         )
#         temp_file_path = temp_file.name
#
#     try:
#         loader = TextLoader(temp_file_path)
#         documents = loader.load()
#
#         print(f"Loaded {len(documents)} document(s)")
#         print(f"Content Preview: {documents[0].page_content[:100]}...")
#         print(f"Metadata: {documents[0].metadata}")
#
#     finally:
#         os.remove(temp_file_path)


def pdf_loader(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s) from PDF")

    for i, doc in enumerate(documents):
        print(f"\nDocument {i + 1}")
        print(f"Content Preview: {doc.page_content[:100]}...")
        print(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    # load_text_file()
    pdf_loader("./doc/1.pdf")