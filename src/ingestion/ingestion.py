# # ingestion.py

# import os
# from typing import List
# from pathlib import Path
# import logging
# import torch
# import tiktoken

# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.schema import Document
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class HealthcareDocumentIngestor:

#     def __init__(self, persist_directory: str = "./chroma_db"):

#         self.persist_directory = persist_directory

#         # Detect GPU automatically
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         logger.info(f"Using device: {device}")

#         self.tokenizer = tiktoken.get_encoding("cl100k_base")

#         logger.info("Loading embedding model...")

#         self.embeddings = HuggingFaceEmbeddings(
#             model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
#             model_kwargs={"device": device},
#             encode_kwargs={"normalize_embeddings": True}
#         )

#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=800,
#             chunk_overlap=100,
#             length_function=self.num_tokens_from_string,
#             separators=["\n## ", "\n### ", "\n#### ", "\n", ". ", "! ", "? ", ", ", " ", ""]
#         )

#         logger.info("Chunk size: 800 tokens | Overlap: 100 tokens")


#     def num_tokens_from_string(self, text: str) -> int:

#         try:
#             return len(self.tokenizer.encode(text))
#         except:
#             return len(text) // 4


#     def load_markdown_files(self, directory_path: str) -> List[Document]:

#         logger.info(f"Loading markdown files from {directory_path}")

#         if not os.path.exists(directory_path):

#             logger.warning(f"{directory_path} does not exist, creating it...")
#             os.makedirs(directory_path, exist_ok=True)
#             return []

#         documents = []

#         md_files = list(Path(directory_path).glob("*.md"))

#         logger.info(f"Found {len(md_files)} markdown files")

#         for md_file in md_files:

#             try:
#                 loader = TextLoader(str(md_file), encoding="utf-8")
#                 docs = loader.load()

#                 for doc in docs:
#                     doc.metadata["source_file"] = md_file.name
#                     doc.metadata["source_type"] = "markdown"

#                 documents.extend(docs)

#                 logger.info(f"Loaded: {md_file.name}")

#             except Exception as e:

#                 logger.error(f"Error loading {md_file.name}: {e}")

#         logger.info(f"Total documents loaded: {len(documents)}")

#         return documents


#     def chunk_documents(self, documents: List[Document]) -> List[Document]:

#         if not documents:

#             logger.warning("No documents to chunk")
#             return []

#         logger.info(f"Chunking {len(documents)} documents")

#         chunks = self.text_splitter.split_documents(documents)

#         for i, chunk in enumerate(chunks):

#             chunk.metadata["chunk_id"] = i
#             chunk.metadata["chunk_size_tokens"] = self.num_tokens_from_string(chunk.page_content)

#         logger.info(f"Created {len(chunks)} chunks")

#         return chunks


#     def ingest_documents(self, chunks: List[Document]) -> Chroma:

#         if not chunks:

#             logger.error("No chunks to ingest")
#             return None

#         logger.info(f"Ingesting {len(chunks)} chunks into ChromaDB")

#         vectorstore = Chroma.from_documents(
#             documents=chunks,
#             embedding=self.embeddings,
#             persist_directory=self.persist_directory
#         )

#         vectorstore.persist()

#         logger.info(f"Successfully stored embeddings in {self.persist_directory}")

#         return vectorstore


#     def process_directory(self, directory_path: str):

#         logger.info("=" * 60)
#         logger.info("STARTING DOCUMENT INGESTION")
#         logger.info("=" * 60)

#         documents = self.load_markdown_files(directory_path)

#         if not documents:

#             logger.error("No documents found. Add .md files inside data/raw")
#             return None

#         chunks = self.chunk_documents(documents)

#         vectorstore = self.ingest_documents(chunks)

#         logger.info("=" * 60)
#         logger.info("INGESTION COMPLETE")
#         logger.info("=" * 60)

#         return vectorstore


# def main():

#     ingestor = HealthcareDocumentIngestor("./chroma_db")

#     vectorstore = ingestor.process_directory("data/raw")

#     if vectorstore:

#         query = "What is the significance of positive Babinski sign with muscle atrophy?"

#         logger.info(f"\nTesting retrieval with query: {query}")

#         results = vectorstore.similarity_search_with_score(query, k=3)

#         for i, (doc, score) in enumerate(results):

#             logger.info(f"\nResult {i+1} | Score: {score}")
#             logger.info(f"Source: {doc.metadata.get('source_file')}")
#             logger.info(doc.page_content[:200])


# if __name__ == "__main__":
#     main()










# ingestion.py
import os
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import tiktoken
from pathlib import Path
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthcareDocumentIngestor:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        
        # Detect GPU automatically
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer for accurate counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Use a biomedical embedding model optimized for healthcare
        logger.info("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Chunking strategy: 500-800 tokens with 100 token overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=self.num_tokens_from_string,
            separators=["\n## ", "\n### ", "\n#### ", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )
        
        logger.info(f"Chunk size: 800 tokens, Overlap: 100 tokens")
    
    def num_tokens_from_string(self, text: str) -> int:
        """Returns the number of tokens in a text string."""
        try:
            return len(self.tokenizer.encode(text))
        except:
            return len(text) // 4
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load and split PDF documents"""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata["source"] = file_path
            doc.metadata["source_file"] = os.path.basename(file_path)
            doc.metadata["source_type"] = "pdf"
        
        return self.text_splitter.split_documents(documents)
    
    def load_markdown(self, file_path: str) -> List[Document]:
        """Load and split markdown files"""
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        for doc in documents:
            doc.metadata["source"] = file_path
            doc.metadata["source_file"] = os.path.basename(file_path)
            doc.metadata["source_type"] = "markdown"
        
        return self.text_splitter.split_documents(documents)
    
    def load_text(self, file_path: str) -> List[Document]:
        """Load and split text files (for uploads)"""
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        
        for doc in documents:
            doc.metadata["source"] = file_path
            doc.metadata["source_file"] = os.path.basename(file_path)
            doc.metadata["source_type"] = "upload"  # Mark as uploaded
            doc.metadata["uploaded"] = True
        
        return self.text_splitter.split_documents(documents)
    
    def load_webpage(self, url: str) -> List[Document]:
        """Load and split webpages"""
        from langchain_community.document_loaders import WebBaseLoader
        loader = WebBaseLoader(url)
        documents = loader.load()
        
        for doc in documents:
            doc.metadata["source"] = url
            doc.metadata["source_file"] = url
            doc.metadata["source_type"] = "webpage"
        
        return self.text_splitter.split_documents(documents)
    
    def load_markdown_files(self, directory_path: str) -> List[Document]:
        """Load all markdown files from a directory"""
        logger.info(f"Loading markdown files from {directory_path}")
        
        if not os.path.exists(directory_path):
            logger.warning(f"Directory {directory_path} does not exist")
            return []
        
        documents = []
        md_files = list(Path(directory_path).glob("*.md"))
        
        logger.info(f"Found {len(md_files)} markdown files")
        
        for md_file in md_files:
            try:
                loader = TextLoader(str(md_file), encoding='utf-8')
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata["source_file"] = md_file.name
                    doc.metadata["source_type"] = "markdown"
                
                documents.extend(docs)
                logger.info(f"  Loaded: {md_file.name}")
                
            except Exception as e:
                logger.error(f"  Error loading {md_file.name}: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            logger.warning("No documents to chunk")
            return []
        
        logger.info(f"Chunking {len(documents)} documents")
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size_tokens"] = self.num_tokens_from_string(chunk.page_content)
        
        logger.info(f"Created {len(chunks)} chunks")
        
        if chunks:
            chunk_sizes = [chunk.metadata["chunk_size_tokens"] for chunk in chunks]
            logger.info(f"Chunk size - Min: {min(chunk_sizes)}, Max: {max(chunk_sizes)}, Avg: {sum(chunk_sizes)/len(chunk_sizes):.0f}")
        
        return chunks
    
    def ingest_documents(self, chunks: List[Document]) -> Chroma:
        """Store chunks in vector database"""
        if not chunks:
            logger.error("No chunks to ingest")
            return None
        
        logger.info(f"Ingesting {len(chunks)} chunks into ChromaDB")
        
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Persist to disk
        vectorstore.persist()
        logger.info(f"✅ Successfully stored embeddings in {self.persist_directory}")
        
        return vectorstore
    
    def process_directory(self, directory_path: str) -> Chroma:
        """Complete pipeline: load, chunk, ingest"""
        logger.info("="*60)
        logger.info("STARTING DOCUMENT INGESTION")
        logger.info("="*60)
        
        # Load documents
        documents = self.load_markdown_files(directory_path)
        
        if not documents:
            logger.error("No documents found")
            return None
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Ingest to vectorstore
        vectorstore = self.ingest_documents(chunks)
        
        logger.info("="*60)
        logger.info("✅ INGESTION COMPLETE")
        logger.info("="*60)
        
        return vectorstore


def main():
    """Main execution function"""
    ingestor = HealthcareDocumentIngestor(persist_directory="./chroma_db")
    
    # Process all markdown files in data/raw
    vectorstore = ingestor.process_directory("data/raw")
    
    if vectorstore:
        # Test retrieval with a sample question
        test_query = "What is the significance of positive Babinski sign with muscle atrophy?"
        logger.info(f"\n🔍 Testing retrieval with query: '{test_query}'")
        
        # Retrieve relevant chunks
        results = vectorstore.similarity_search_with_score(test_query, k=3)
        
        logger.info(f"\nTop 3 results:")
        for i, (doc, score) in enumerate(results):
            logger.info(f"\n--- Result {i+1} (Score: {score:.4f}) ---")
            logger.info(f"Source: {doc.metadata.get('source_file', 'Unknown')}")
            logger.info(f"Content: {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()