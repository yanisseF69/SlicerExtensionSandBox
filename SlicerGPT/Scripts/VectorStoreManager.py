# pip install langchain langchain_community langchain_huggingface faiss-cpu
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class VectorStoreManager:
    def __init__(self, index_root: str, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the vector store manager.

        Args:
            index_root (str): Path to the directory containing all FAISS sub-indexes.
            embedding_model (str): HuggingFace model name used to generate embeddings.
        """
        self.index_root = index_root
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.index = None
        self.load_and_merge_indexes()

    def load_and_merge_indexes(self):
        """
        Load all FAISS sub-indexes from the specified directory and merge them into a single index.
        """
        index_dirs = [
            os.path.join(self.index_root, d) 
            for d in os.listdir(self.index_root) 
            if os.path.isdir(os.path.join(self.index_root, d))
        ]

        if not index_dirs:
            raise ValueError(f"No FAISS indexes found in {self.index_root}")

        # Load the first index
        self.index = FAISS.load_local(index_dirs[0], self.embeddings, allow_dangerous_deserialization=True)

        # Merge the remaining indexes
        for sub_index_dir in index_dirs[1:]:
            sub_index = FAISS.load_local(sub_index_dir, self.embeddings, allow_dangerous_deserialization=True)
            self.index.merge_from(sub_index)

    def search(self, query: str, k: int = 3):
        """
        Perform a similarity search on the merged index.

        Args:
            query (str): The text query to search for.
            k (int): Number of top results to return.

        Returns:
            List[Document]: The top-k most similar documents.
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call `load_and_merge_indexes()` first.")
        return self.index.similarity_search(query, k=k)

    def save_merged_index(self, path: str):
        """
        Save the merged FAISS index to a specified directory.

        Args:
            path (str): Path to the output directory where the index will be saved.
        """
        if self.index is None:
            raise RuntimeError("No index to save. Call `load_and_merge_indexes()` first.")
        self.index.save_local(path)
        print(f"Merged index saved to: {path}")

if __name__ == "__main__":
    manager = VectorStoreManager("./SlicerFAISS/")
    manager.load_and_merge_indexes()

    results = manager.search("How to write a scripted module in 3D Slicer?", k=3)
    for doc in results:
        print(doc.page_content)