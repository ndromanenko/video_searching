import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document


class Retrieval:
    def __init__(self, k: int = 10) -> None:
        """
        Initialize Retrieval with OpenAI embeddings and FAISS index.

        Args:
            k: Number of similar documents to retrieve.
            
        """
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        self.retriever: FAISS = FAISS(
            index=self.index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
            embedding_function=self.embeddings
        )
        self.k: int = k

    def add_texts(self, texts: list[str], metadata: list[dict[str, str]]) -> None:
        """
        Add texts and their metadata to the retriever.

        Args:
            texts: List of texts to add.
            metadata: List of metadata dictionaries for each text.
            
        """
        self.retriever.add_texts(texts=texts, metadatas=metadata)

    def search(self, query: str, filter_criteria: dict[str, str] | None = None) -> list[Document]:
        """
        Search for similar documents.

        Args:
            query: Text to search for.
            filter_criteria: Optional filter criteria.

        Returns:
            List of Document objects containing the search results.
            
        """
        return self.retriever.similarity_search(query=query, k=self.k, filter=filter_criteria)
