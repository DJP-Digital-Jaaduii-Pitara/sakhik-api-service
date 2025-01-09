import argparse
import json
from typing import (
    Dict,
    List
)

import marqo
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import SimpleDirectoryReader


def load_documents(folder_path, input_chunk_size, input_chunk_overlap):
    source_chunks = []
    sources = SimpleDirectoryReader(
        input_dir=folder_path, recursive=True).load_data()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=input_chunk_size, chunk_overlap=input_chunk_overlap)
    for source in sources:
        for chunk in splitter.split_text(source.text):
            source_chunks.append(Document(page_content=chunk, metadata={
                "page_label": source.metadata.get("page_label"),
                "file_name": source.metadata.get("file_name"),
                "file_path": source.metadata.get("file_path"),
                "file_type": source.metadata.get("file_type")
            }))
    return source_chunks


def get_formatted_documents(documents: List[Document]):
    docs: List[Dict[str, str]] = []
    for d in documents:
        doc = {
            "text": d.page_content,
            "metadata": json.dumps(d.metadata) if d.metadata else json.dumps({}),
        }
        docs.append(doc)
    return docs


def chunk_list(document, batch_size):
    """Return a list of batch sized chunks from document."""
    return [document[i: i + batch_size] for i in range(0, len(document), batch_size)]


def indexer_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--marqo_url',
                        type=str,
                        required=True,
                        help='Endpoint URL of marqo',
                        )
    parser.add_argument('--index_name',
                        type=str,
                        required=True,
                        help='Name of marqo index',
                        )
    parser.add_argument('--folder_path',
                        type=str,
                        required=True,
                        help='Path to the folder',
                        default="input_data"
                        )
    parser.add_argument('--embedding_model',
                        type=str,
                        required=False,
                        help='data embedding model to be used.',
                        default="flax-sentence-embeddings/all_datasets_v4_mpnet-base"
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
    parser.add_argument('--split_length',
                        type=int,
                        required=False,
                        help='pre-processing split_length',
                        default=1
                        )
    parser.add_argument('--split_overlap',
                        type=int,
                        required=False,
                        help='pre-processing split_overlap',
                        default=0
                        )
    parser.add_argument('--fresh_index',
                        action='store_true',
                        help='Is the indexing fresh'
                        )

    args = parser.parse_args()

    MARQO_URL = args.marqo_url
    MARQO_INDEX_NAME = args.index_name
    FOLDER_PATH = args.folder_path
    FRESH_INDEX = args.fresh_index
    EMBED_MODEL = args.embedding_model
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    SPLIT_LENGTH = args.split_length
    SPLIT_OVERLAP = args.split_overlap

    index_settings = {
                "treatUrlsAndPointersAsImages": False,
                "model": EMBED_MODEL,
                "normalizeEmbeddings": True,
                "textPreprocessing": {
                    "splitLength": SPLIT_LENGTH,
                    "splitOverlap": SPLIT_OVERLAP,
                    "splitMethod": "passage"
                }
            }


    # Initialize Marqo instance
    marqo_client = marqo.Client(url=MARQO_URL)
    if FRESH_INDEX:
        try:
            marqo_client.index(MARQO_INDEX_NAME).delete()
            print("Existing Index successfully deleted.")
        except:
            print("Index does not exist. Creating new index")

        marqo_client.create_index(
            MARQO_INDEX_NAME, settings_dict=index_settings)
        print(f"Index {MARQO_INDEX_NAME} created.")

    print("Loading documents...")
    documents = load_documents(FOLDER_PATH, CHUNK_SIZE, CHUNK_OVERLAP)

    print("Total Documents ===>", len(documents))

    f = open("indexed_documents.txt", "w")
    f.write(str(documents))
    f.close()

    print(f"Indexing documents...")
    formatted_documents = get_formatted_documents(documents)
    tensor_fields = ['text']
    _document_batch_size = 50
    chunks = list(chunk_list(formatted_documents, _document_batch_size))
    for chunk in chunks:
        marqo_client.index(MARQO_INDEX_NAME).add_documents(
            documents=chunk, client_batch_size=_document_batch_size, tensor_fields=tensor_fields)

    print("============ INDEX DONE =============")


if __name__ == "__main__":
    indexer_main()
    
# RUN
# python3 index_documents.py --marqo_url=http://0.0.0.0:8882 --index_name=sakhi_activity --folder_path=/home/ttpl-rt-057/Documents/dpdpa --fresh_index (FOR FRESH INDEXING)
# python3 index_documents.py --marqo_url=http://0.0.0.0:8882 --index_name=sakhi_activity --folder_path=input_data (FOR APPENDING DOCUMENTS)
