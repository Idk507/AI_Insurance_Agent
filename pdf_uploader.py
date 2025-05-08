import weaviate
from weaviate.auth import AuthApiKey
import pdfplumber
from openai import AzureOpenAI
import os

# Configuration settings
# WEAVIATE_CONFIG = {
#     "http_host": "localhost",
#     "http_port": 8080,
#     "grpc_host": "localhost",
#     "grpc_port": 50051,
#     "secure": False
# }

WEAVIATE_CONFIG = {
    "url": "0q009z7qqsgnjfbhatmbq.c0.asia-southeast1.gcp.weaviate.cloud",
    "api_key": "WKMXWnRlTBXsvOA1OBhfnGDiPzTpXAaWc5T3",
    "class_name": "singaporecluster"
}


AZURE_CONFIG = {
    "api_key":  "75bGVIjxjnTYUbcK1RAhTxqgIfrZPDXRVZQHAwsGxmQ7xC2Cx4zfJQQJ99BEACYeBjFXJ3w3AAABACOGimGC",
    "endpoint": "https://gptidk.openai.azure.com/",
    "api_version": "2024-12-01-preview",
    "embedding_deployment": "text-embedding-ada-002",
    "gpt_deployment": "gptidk"
}

class PDFUploader:
    def __init__(self, weaviate_config, azure_config):
        # self.client = weaviate.connect_to_custom(
        #     http_host=weaviate_config["http_host"],
        #     http_port=weaviate_config["http_port"],
        #     grpc_host=weaviate_config["grpc_host"],
        #     grpc_port=weaviate_config["grpc_port"],
        #     http_secure=weaviate_config["secure"],
        #     grpc_secure=weaviate_config["secure"]
        # )
        
        self.client = weaviate.connect_to_wcs(
                cluster_url=WEAVIATE_CONFIG["url"],
                auth_credentials=AuthApiKey(WEAVIATE_CONFIG["api_key"]),
            )

        self.azure_client = None
        if azure_config:
            self.azure_client = AzureOpenAI(
                azure_endpoint=azure_config["endpoint"],
                api_key=azure_config["api_key"],
                api_version=azure_config["api_version"]
            )
            self.embedding_model = azure_config["embedding_deployment"]
    
    def create_collection(self, collection_name, description):
        # Check if collection already exists
        if self.client.collections.exists(collection_name):
            print(f"Collection '{collection_name}' already exists.")
            return

        properties = [
            weaviate.classes.config.Property(
                name="title",
                data_type=weaviate.classes.config.DataType.TEXT,
                description="Title of the document"
            ),
            weaviate.classes.config.Property(
                name="content",
                data_type=weaviate.classes.config.DataType.TEXT,
                description="Content of the document"
            )
        ]
        
        self.client.collections.create(
            name=collection_name,
            description=description,
            properties=properties,
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.none()
        )
        print(f"Collection '{collection_name}' created successfully.")
    
    def add_pdf_to_collection(self, collection_name, pdf_path):
        if not self.client.collections.exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
        # Extract text from PDF
        with pdfplumber.open(pdf_path) as pdf:
            text_content = "\n".join(
                [page.extract_text() for page in pdf.pages if page.extract_text()]
            )
        
        collection = self.client.collections.get(collection_name)
        embedding = self.get_text_embedding(text_content) if self.azure_client else None
        
        # Insert data into collection
        collection.data.insert(
            properties={
                "title": os.path.basename(pdf_path),
                "content": text_content,
            },
            vector=embedding
        )
        
        print(f"PDF '{pdf_path}' added to collection '{collection_name}'.")
        return {
            "title": os.path.basename(pdf_path),
            "status": "success"
        }
    
    def get_text_embedding(self, text):
        if not self.azure_client:
            raise ValueError("Azure OpenAI client not configured.")
        
        response = self.azure_client.embeddings.create(
            input=text, 
            model=self.embedding_model
        )
        
        return response.data[0].embedding
    
    def list_all_documents(self, collection_name):
        if not self.client.collections.exists(collection_name):
            return []
        
        collection = self.client.collections.get(collection_name)
        results = collection.query.fetch_objects(
            return_properties=["title", "content"]
        )
        
        documents = []
        for result in results.objects:
            documents.append({
                "title": result.properties['title'],
                "id": result.uuid
            })
        
        return documents


def batch_upload_pdfs(directory, collection_name="InsuranceDocs"):
    """Utility function to upload all PDFs from a directory"""
    uploader = PDFUploader(WEAVIATE_CONFIG, AZURE_CONFIG)
    
    # Create collection if it doesn't exist
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
            print(f"Skipping non-PDF file: {filename}")
    
    return results


if __name__ == "__main__":
    # Example usage
    dir_path = "Major Hong Leong Assurance Plans"
    
    if not os.path.exists(dir_path):
        print(f"Directory '{dir_path}' does not exist.")
    else:
        results = batch_upload_pdfs(dir_path)
        print(f"Uploaded {len(results)} documents")
        
        # List all documents
        uploader = PDFUploader(WEAVIATE_CONFIG, AZURE_CONFIG)
        docs = uploader.list_all_documents("InsuranceDocs")
        print(f"Total documents in collection: {len(docs)}")
        for doc in docs:
            print(f"- {doc['title']}")
            