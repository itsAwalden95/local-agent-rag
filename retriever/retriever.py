from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List, Any
import logging
import os
from dotenv import load_dotenv
from pathlib import Path
from utils.utils import config
import pypandoc
from PyPDF2 import PdfReader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.retrievers import EnsembleRetriever, BM25Retriever
import pickle

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles document conversion and splitting using pypandoc and PDF specific processing.
    """
    def __init__(self, headers_to_split_on: List[str]):
        self.headers_to_split_on = headers_to_split_on

    def extract_text_from_pdf(self, filepath: str) -> str:
        """
        Extracts text from PDF files using PyPDF2.

        Args:
            filepath (str): Path to the PDF file

        Returns:
            str: Extracted text content
        """
        try:
            text = []
            with open(filepath, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())

            return '\n\n'.join(text)

        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise RuntimeError(f"Error extracting text from PDF: {e}")

    def convert_to_markdown(self, filepath: str) -> str:
        """
        Converts various document formats to markdown using pypandoc.
        Supports: docx, pdf, html, odt, and more.
        
        Args:
            filepath (str): Path to the source document
            
        Returns:
            str: Markdown formatted text
        """
        try:
            file_extension = Path(filepath).suffix.lower()
            
            # Handle PDF files separately
            if file_extension == '.pdf':
                # Extract text from PDF
                text_content = self.extract_text_from_pdf(filepath)
                
                # Convert extracted text to markdown using pypandoc
                markdown_content = pypandoc.convert_text(
                    text_content,
                    'markdown',
                    format='commonmark',
                    extra_args=['--wrap=none']
                )
                return markdown_content
            
            # For non-PDF files, use standard pypandoc conversion
            # Define input format based on file extension
            format_mapping = {
                '.docx': 'docx',
                '.html': 'html',
                '.odt': 'odt',
                '.txt': 'text',
                '.md': 'markdown'
            }
            
            input_format = format_mapping.get(file_extension, 'text')
            
            # Convert to markdown
            markdown_content = pypandoc.convert_file(
                filepath,
                'markdown',
                format=input_format,
                extra_args=['--wrap=none']
            )
            
            logger.info(f"Successfully converted {filepath} to markdown")
            return markdown_content
            
        except Exception as e:
            logger.error(f"Error converting document to markdown: {e}")
            raise RuntimeError(f"Error converting document: {e}")

    def process(self, source: str) -> List[str]:
        """
        Converts a document to markdown and splits it into chunks.

        Args:
            source (str): The path to the source document.

        Returns:
            List[str]: List of document sections split by headers.
        """
        try:
            logger.info("Starting document processing.")
            
            # Convert document to markdown
            markdown_document = self.convert_to_markdown(source)
            
            # Split the markdown document
            markdown_splitter = MarkdownHeaderTextSplitter(self.headers_to_split_on)
            docs_list = markdown_splitter.split_text(markdown_document)
            
            logger.info("Document processed successfully.")
            return docs_list
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise RuntimeError(f"Error processing document: {e}")
class IndexBuilder:
    def __init__(self, docs_list: List[str], collection_name: str, persist_directory: str, load_documents: bool):
        self.docs_list = docs_list
        self.collection_name = collection_name
        self.vectorstore = None
        self.persist_directory = persist_directory
        self.load_documents = load_documents

    def build_vectorstore(self):
        """
        Initializes the FAISS vectorstore with the provided documents and embeddings.
        """
        try:
            logger.info("Initializing local embeddings model.")
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            logger.info("Building vectorstore.")
            
            # Create FAISS index from documents
            self.vectorstore = FAISS.from_documents(
                self.docs_list,
                embeddings
            )
            
            # Ensure directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Save the FAISS index
            save_path = os.path.join(self.persist_directory, self.collection_name)
            self.vectorstore.save_local(save_path)
            
            logger.info(f"Vectorstore built and saved successfully at {save_path}")
            
        except Exception as e:
            logger.error(f"Error building vectorstore: {e}")
            raise RuntimeError(f"Error building vectorstore: {e}")

    def build_retrievers(self):
        """
        Builds BM25 and vector-based retrievers and combines them into an ensemble retriever.
        """
        try:
            logger.info("Building BM25 retriever.")
            bm25_retriever = BM25Retriever.from_documents(self.docs_list, search_kwargs={"k": 4})

            logger.info("Building vector-based retrievers.")
            retriever_vanilla = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            retriever_mmr = self.vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": 4}
            )

            logger.info("Combining retrievers into an ensemble retriever.")
            ensemble_retriever = EnsembleRetriever(
                retrievers=[retriever_vanilla, retriever_mmr, bm25_retriever],
                weights=[0.3, 0.3, 0.4],
            )
            logger.info("Retrievers built successfully.")
            return ensemble_retriever
        except Exception as e:
            logger.error(f"Error building retrievers: {e}")
            raise RuntimeError(f"Error building retrievers: {e}")

if __name__ == "__main__":
    # Configuration
    headers_to_split_on = config["retriever"]["headers_to_split_on"]
    filepath = config["retriever"]["file"]
    collection_name = config["retriever"]["collection_name"]
    load_documents = config["retriever"]["load_documents"]

    print("Retriever entry")
    if load_documents:
        # Document Processing
        logger.info("Initializing document processor.")
        processor = DocumentProcessor(headers_to_split_on)
        try:        
            docs_list = processor.process(filepath)    
            logger.info(f"{len(docs_list)} chunks generated.") 
        except RuntimeError as e:        
            logger.info(f"Failed to process document: {e}")        
            exit(1)

        # Index Building
        logger.info("Initializing index builder.")
        index_builder = IndexBuilder(docs_list, collection_name, persist_directory="vector_db", load_documents=load_documents)
        index_builder.build_vectorstore()

        try:
            ensemble_retriever = index_builder.build_retrievers()
            logger.info("Index and retrievers built successfully. Ready for use.")
        except RuntimeError as e:
            logger.critical(f"Failed to build index or retrievers: {e}")
            exit(1)
