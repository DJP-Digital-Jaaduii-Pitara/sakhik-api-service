import json
import os
from typing import (
    Dict,
    List,
    Tuple
)

import marqo
from langchain.docstore.document import Document
from langchain.vectorstores.marqo import Marqo

from vectorstores.base import BaseVectorStore


class MarqoVectorStore(BaseVectorStore):
    TENSOR_FIELDS: str = ["text"]
    client: marqo.Client

    def __init__(self):
        self.client_url = os.environ["VECTOR_STORE_ENDPOINT"]
        self.collection_name = os.environ["VECTOR_COLLECTION_NAME"]
        self.embedding_model = os.environ["EMBEDDING_MODEL"]
        self.index_settings = {
            "treatUrlsAndPointersAsImages": False,
            "model": self.embedding_model,
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": self.SPLIT_LENGTH,
                "splitOverlap": self.SPLIT_OVERLAP,
                "splitMethod": "passage"
            }
        }

        if not self.client_url:
            raise ValueError("Missing environment variable VECTOR_STORE_ENDPOINT.")

        if not self.collection_name:
            raise ValueError("Missing environment variable VECTOR_COLLECTION_NAME.")

        if not self.embedding_model:
            raise ValueError("Missing environment variable EMBEDDING_MODEL.")

        self.client = marqo.Client(url=self.client_url)

    def get_client(self) -> marqo.Client:
        return self.client

    def add_documents(self, documents=List[Document], fresh_collection: bool = False) -> List[str]:

        if fresh_collection:
            try:
                self.client.index(self.collection_name).delete()
                print("Existing Index successfully deleted.")
            except:
                print("Index does not exist. Creating new index...")

            self.client.create_index(
                self.collection_name, settings_dict=self.index_settings)
            print(f"Index {self.collection_name} created.")

        docs: List[Dict[str, str]] = []
        ids = []
        for d in documents:
            doc = {
                "text": d.page_content,
                "metadata": json.dumps(d.metadata) if d.metadata else json.dumps({}),
            }
            docs.append(doc)
        chunks = list(self.chunk_list(docs, self.BATCH_SIZE))
        for chunk in chunks:
            response = self.client.index(self.collection_name).add_documents(
                documents=chunk, client_batch_size=self.BATCH_SIZE, tensor_fields=self.TENSOR_FIELDS)
            if response[0]["errors"]:
                print(f'>>>>> {response}')
                err_msg = (
                    f"Error in upload for documents in index range"
                    f"check Marqo logs."
                )
                raise RuntimeError(err_msg)
            ids += [item["_id"] for item in response[0]["items"]]

        return ids

    def similarity_search_with_score(self, query: str, collection_name: str, k: int = 20) -> List[Tuple[Document, float]]:
        docsearch = Marqo(self.client, index_name=collection_name)
        documents = docsearch.similarity_search_with_score(query, k)
        return documents