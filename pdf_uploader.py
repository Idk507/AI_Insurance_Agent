import weaviate
import pdfplumber
from openai import AzureOpenAI
import os
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.init import Auth
import logging
from typing import List, Dict, Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration settings
WEAVIATE_CONFIG = {
    "http_host": "localhost",
    "http_port": 8080,
    "grpc_host": "localhost",
    "grpc_port": 50051,
    "secure": False
}

AZURE_CONFIG =  {
    "api_key": "02j8yI53TMGAgFEoDuiCkrYuUzXLB5N2hJik92GyFvnnSxPt68uHJQQJ99BAACYeBjFXJ3w3AAABACOG4ocs",
    "endpoint": "https://genairag.openai.azure.com/",
    "api_version": "2023-05-15",
    "embedding_deployment": "text-embedding-ada-002",
    "gpt_deployment": "gpt-4o"
}

class PDFUploader:
    def __init__(self, weaviate_config: Dict, azure_config: Dict):
        try:
            self.client = weaviate.connect_to_custom(
                http_host=weaviate_config["http_host"],
                http_port=weaviate_config["http_port"],
                grpc_host=weaviate_config["grpc_host"],
                grpc_port=weaviate_config["grpc_port"],
                http_secure=weaviate_config["secure"],
                grpc_secure=weaviate_config["secure"],
                
            )
            logger.info("Connected to Weaviate successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            raise

        self.azure_client = None
        if azure_config:
            try:
                self.azure_client = AzureOpenAI(
                    azure_endpoint=azure_config["endpoint"],
                    api_key=azure_config["api_key"],
                    api_version=azure_config["api_version"]
                )
                self.embedding_model = azure_config["embedding_deployment"]
                logger.info("Initialized Azure OpenAI client")
            except Exception as e:
                logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
                raise

    def create_collection(self, collection_name: str, description: str) -> None:
        if self.client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists")
            return

        try:
            properties = [
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="Title of the document"
                ),
                Property(
                    name="content",
                    data_type=DataType.TEXT,
                    description="Content of the document"
                )
            ]

            self.client.collections.create(
                name=collection_name,
                description=description,
                properties=properties,
                vectorizer_config=Configure.Vectorizer.none(),  # Use external embeddings
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric="cosine",
                    vector_cache_max_objects=100000
                )
            )
            logger.info(f"Collection '{collection_name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {str(e)}")
            raise

    def add_pdf_to_collection(self, collection_name: str, pdf_path: str) -> Dict:
        if not self.client.collections.exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        try:
            # Extract text from PDF
            text_content = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            if not text_content:
                raise ValueError("No text could be extracted from the PDF")

            collection = self.client.collections.get(collection_name)
            embedding = self.get_text_embedding(text_content) if self.azure_client else None

            # Insert data into collection
            response = collection.data.insert(
                properties={
                    "title": os.path.basename(pdf_path),
                    "content": text_content,
                },
                vector=embedding
            )

            logger.info(f"PDF '{pdf_path}' added to collection '{collection_name}'")
            return {
                "title": os.path.basename(pdf_path),
                "status": "success",
                "uuid": str(response)
            }
        except Exception as e:
            logger.error(f"Failed to add PDF '{pdf_path}' to collection '{collection_name}': {str(e)}")
            raise

    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        if not self.azure_client:
            raise ValueError("Azure OpenAI client not configured")

        try:
            # Truncate text if too long (ada-002 has ~8191 token limit)
            max_length = 8000  # Conservative character limit
            if len(text) > max_length:
                text = text[:max_length]
                logger.warning("Text truncated to %d characters for embedding", max_length)

            response = self.azure_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            if len(embedding) != 1536:
                raise ValueError(f"Expected 1536-dimensional embedding, got {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise

    def list_all_documents(self, collection_name: str) -> List[Dict]:
        if not self.client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' does not exist")
            return []

        try:
            collection = self.client.collections.get(collection_name)
            results = collection.query.fetch_objects(
                return_properties=["title", "content"]
            )

            documents = [
                {
                    "title": result.properties['title'],
                    "id": str(result.uuid)
                }
                for result in results.objects
            ]
            logger.info(f"Retrieved {len(documents)} documents from collection '{collection_name}'")
            return documents
        except Exception as e:
            logger.error(f"Failed to list documents in collection '{collection_name}': {str(e)}")
            raise

    # def __del__(self):
    #     if hasattr(self, 'client') and self.client:
    #         self.client.close()
    #         logger.info("Weaviate client closed")

def batch_upload_pdfs(directory: str, collection_name: str = "InsuranceDocs") -> List[Dict]:
    """Utility function to upload all PDFs from a directory"""
    uploader = PDFUploader(WEAVIATE_CONFIG, AZURE_CONFIG)
    
    uploader.create_collection(
        collection_name,
        "Collection for storing insurance PDF contents"
    )
    
    results = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(directory, filename)
            try:
                result = uploader.add_pdf_to_collection(collection_name, pdf_path)
                results.append(result)
            except Exception as e:
                results.append({
                    "title": filename,
                    "status": "error",
                    "message": str(e)
                })
        else:
            logger.info(f"Skipping non-PDF file: {filename}")
    
    return results

# if __name__ == "__main__":
#     dir_path = "Major Hong Leong Assurance Plans"
    
#     if not os.path.exists(dir_path):
#         logger.error(f"Directory '{dir_path}' does not exist")
#     else:
#         results = batch_upload_pdfs(dir_path)
#         logger.info(f"Uploaded {len(results)} documents")
        
#         uploader = PDFUploader(WEAVIATE_CONFIG, AZURE_CONFIG)
#         docs = uploader.list_all_documents("InsuranceDocs")
#         logger.info(f"Total documents in collection: {len(docs)}")
#         for doc in docs:
#             logger.info(f"- {doc['title']}")