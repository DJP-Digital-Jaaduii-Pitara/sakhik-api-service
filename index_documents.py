import argparse
import re
from typing import (
    List
)
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import SimpleDirectoryReader
from env_manager import vectorstore_class
 
def document_loader(input_dir: str) -> List[Document]:
    """Load data from the input directory.

    Args:
        input_dir (str): Path to the directory.

    Returns:
        List[Document]: A list of documents.
    """
    return SimpleDirectoryReader(
        input_dir=input_dir, recursive=True).load_data() # show_progress=True 
 
def split_documents(documents: List[Document], chunk_size: int = 4000, chunk_overlap = 200) -> List[Document]:
    """Split documents.

    Args:
        documents: List of documents
        chunk_size: Maximum size of chunks to return
        chunk_overlap: Overlap in characters between chunks

    Returns:
        List[Document]: A list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # splited_docs = text_splitter.split_documents(documents)
    pattern = re.compile(r'[\x00-\x1F\x7F\u00A0]')
    splited_docs = []
    for document in documents:
        for chunk in text_splitter.split_text(document.text):
            chunk = re.sub(pattern, '', chunk)
            splited_docs.append(Document(page_content=chunk, metadata={
                "page_label": document.metadata.get("page_label"),
                "file_name": document.metadata.get("file_name"),
                # "file_path": document.metadata.get("file_path"),
                "file_type": document.metadata.get("file_type")
            }))
    return splited_docs

def transform_documents():
    pass

def load_documents(folder_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    documents = document_loader(folder_path)
    splitted_documents = split_documents(documents, chunk_size, chunk_overlap)
    return splitted_documents

def indexer_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path',
                        type=str,
                        required=True,
                        help='Path to the folder',
                        default="input_data"
                        )
    parser.add_argument('--chunk_size',
                        type=int,
                        required=False,
                        help='documents chunk size',
                        default=1024
                        )
    parser.add_argument('--chunk_overlap',
                        type=int,
                        required=False,
                        help='documents chunk size',
                        default=200
                        )
    parser.add_argument('--fresh_index',
                        action='store_true',
                        help='Is the indexing fresh'
                        )

    args = parser.parse_args()

    FOLDER_PATH = args.folder_path
    FRESH_INDEX = args.fresh_index
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap

    documents = load_documents(FOLDER_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    print("Total documents :: =>", len(documents))
    
    print("Adding documents...")
    results = vectorstore_class.add_documents(documents, FRESH_INDEX)
    print("results =======>", results)
    
    print("============ INDEX DONE =============")


if __name__ == "__main__":
    indexer_main()
    
# For Fresh collection
# python3 index_documents.py --folder_path=Documents --fresh_index

# For appending documents to existing collection
# python3 index_documents.py --folder_path=Documents