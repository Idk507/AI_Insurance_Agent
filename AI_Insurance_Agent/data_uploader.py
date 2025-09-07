import weaviate
from weaviate.auth import AuthApiKey
import pdfplumber
from openai import AzureOpenAI
import os
import json
import hashlib
from filelock import FileLock
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
import logging
from typing import Dict, Optional, List
from langchain.memory import ConversationBufferMemory
import csv

# Setup logging
logging.basicConfig(
    filename='insurance_assistant.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Configuration settings
AZURE_CONFIG = {
    "api_key": "75bGVIjxjnTYUbcK1RAhTxqgIfrZPDXRVZQHAwsGxmQ7xC2Cx4zfJQQJ99BEACYeBjFXJ3w3AAABACOGimGC",
    "endpoint": "https://gptidk.openai.azure.com/",
    "api_version": "2024-12-01-preview",
    "embedding_deployment": "text-embedding-ada-002",
    "gpt_deployment": "gpt-4o"
}

WEAVIATE_URL = "nwhexgobshomixnhqbhlg.c0.asia-southeast1.gcp.weaviate.cloud"
WEAVIATE_API_KEY = "STV3MHJ1ZEl0OTJaTDJaY182VXRuSXVTUlQ1UWxDKzFTRGgrNlFsdlpnMzlVeGxWMkYwemFDdHh0dC9FPV92MjAw"
collection_names = ["CustomerProfiles", "InsuranceDocs"]
WEAVIATE_CONFIG = {
    "url": WEAVIATE_URL,
    "api_key": WEAVIATE_API_KEY,
    "class_name": "singaporeclusters",
    "collection_names": collection_names
}




class DataUploader:
    def __init__(self, weaviate_config: Dict, azure_config: Dict):
        """Initialize Weaviate and Azure OpenAI clients."""
        try:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_config["url"],
                auth_credentials=AuthApiKey(weaviate_config["api_key"]),
            )
            logger.info("Connected to Weaviate successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            raise

        self.azure_client = None
        self.embedding_model = None
        if azure_config and azure_config.get("api_key") and azure_config.get("endpoint"):
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
                self.azure_client = None
        else:
            logger.warning("Azure OpenAI configuration incomplete. Embedding generation will be skipped.")

        self.processed_files_path = os.path.join('customer_data', 'processed_files.json')
        os.makedirs('customer_data', exist_ok=True)
        self.lock = FileLock(self.processed_files_path + '.lock')
        self.collection_names = weaviate_config["collection_names"]
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    def _load_processed_files(self) -> Dict:
        with self.lock:
            if os.path.exists(self.processed_files_path):
                try:
                    with open(self.processed_files_path, 'r') as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted {self.processed_files_path}, starting fresh")
                    return {}
            return {}

    def _save_processed_files(self, processed_files: Dict) -> None:
        with self.lock:
            with open(self.processed_files_path, 'w') as f:
                json.dump(processed_files, f, indent=2)
            logger.info(f"Updated processed files tracker at {self.processed_files_path}")

    def create_collection(self, collection_name: str, description: str) -> None:
        if self.client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists")
            return

        try:
            properties = [
                Property(name="title", data_type=DataType.TEXT, description="Title of the document (Customer ID or File Name)"),
                Property(name="content", data_type=DataType.TEXT, description="Content of the document (JSON string for customer profiles or text for insurance documents)"),
                Property(name="category", data_type=DataType.TEXT, description="Category: customer_profile or insurance_document")
            ]

            self.client.collections.create(
                name=collection_name,
                description=description,
                properties=properties,
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    vector_cache_max_objects=100000
                )
            )
            logger.info(f"Collection '{collection_name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {str(e)}")
            raise

    def add_customer_profile_csv(self, collection_name: str, csv_path: str) -> Dict:
        if not self.client.collections.exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        file_hash = self.get_file_hash(csv_path)
        processed_files = self._load_processed_files()

        try:
            if not os.path.isfile(csv_path):
                raise ValueError(f"CSV file not found: {csv_path}")
            if not os.access(csv_path, os.R_OK):
                raise ValueError(f"CSV file not readable: {csv_path}")

            customer_details_list = self.extract_customer_details_from_csv(csv_path)
            if not customer_details_list:
                logger.warning(f"No customer profiles extracted from '{csv_path}'")
                processed_files[file_hash] = {
                    "path": csv_path,
                    "category": "customer_profile",
                    "profiles_extracted": 0
                }
                self._save_processed_files(processed_files)
                return {"title": os.path.basename(csv_path), "status": "failed", "message": "No profiles extracted"}

            collection = self.client.collections.get(collection_name)
            inserted_profiles = 0
            for customer_details in customer_details_list:
                try:
                    customer_id = customer_details.get("Customer ID", f"CUST{hashlib.md5(json.dumps(customer_details).encode()).hexdigest()[:8]}")
                    details_json = json.dumps(customer_details)
                    properties = {
                        "title": customer_id,
                        "content": details_json,
                        "category": "customer_profile"
                    }
                    embedding = None
                    if self.azure_client:
                        try:
                            embedding = self.get_text_embedding(details_json)
                        except Exception as e:
                            logger.warning(f"Embedding failed for '{customer_id}': {str(e)}. Proceeding without embedding.")
                    collection.data.insert(properties=properties, vector=embedding)
                    inserted_profiles += 1
                except Exception as e:
                    logger.error(f"Failed to insert profile for '{customer_id}': {str(e)}")
                    continue

            processed_files[file_hash] = {
                "path": csv_path,
                "category": "customer_profile",
                "profiles_extracted": inserted_profiles
            }
            self._save_processed_files(processed_files)

            logger.info(f"Customer profiles from '{csv_path}' added to collection '{collection_name}'")
            return {
                "title": os.path.basename(csv_path),
                "status": "success",
                "profiles_extracted": inserted_profiles
            }
        except Exception as e:
            logger.error(f"Failed to add customer profile CSV '{csv_path}' to collection '{collection_name}': {str(e)}")
            raise

    def add_insurance_document_pdf(self, collection_name: str, pdf_path: str) -> Dict:
        if not self.client.collections.exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")

        pdf_hash = self.get_file_hash(pdf_path)
        processed_files = self._load_processed_files()

        try:
            if not os.path.isfile(pdf_path):
                raise ValueError(f"PDF file not found: {pdf_path}")
            if not os.access(pdf_path, os.R_OK):
                raise ValueError(f"PDF file not readable: {pdf_path}")

            text_content = self.extract_text_from_pdf(pdf_path)
            if not text_content.strip():
                raise ValueError(f"No text could be extracted from the PDF")

            properties = {
                "title": os.path.basename(pdf_path),
                "content": text_content,
                "category": "insurance_document"
            }

            embedding = None
            if self.azure_client:
                try:
                    embedding = self.get_text_embedding(text_content)
                except Exception as e:
                    logger.warning(f"Embedding failed for '{pdf_path}': {str(e)}. Proceeding without embedding.")

            collection = self.client.collections.get(collection_name)
            response = collection.data.insert(
                properties=properties,
                vector=embedding
            )

            processed_files[pdf_hash] = {
                "path": pdf_path,
                "category": "insurance_document",
                "uuid": str(response),
                "collection": collection_name
            }
            self._save_processed_files(processed_files)

            logger.info(f"Insurance document from '{pdf_path}' added to collection '{collection_name}'")
            return {
                "title": os.path.basename(pdf_path),
                "status": "success",
                "uuid": str(response)
            }
        except Exception as e:
            logger.error(f"Failed to add insurance document PDF '{pdf_path}' to collection '{collection_name}': {str(e)}")
            raise

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        lock_file = f"{pdf_path}.lock"
        with FileLock(lock_file):
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = "".join(page.extract_text() or "" for page in pdf.pages)
                logger.info(f"Extracted {len(text)} characters from PDF '{pdf_path}'")
                return text
            except Exception as e:
                logger.error(f"Failed to extract text from PDF '{pdf_path}': {str(e)}")
                raise

    def extract_customer_details_from_csv(self, csv_path: str) -> List[Dict]:
        """Extract all customer details from a CSV file and return as a list of dictionaries."""
        try:
            customer_details_list = []
            with open(csv_path, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    customer_details = {key: value for key, value in row.items() if value}
                    if customer_details:
                        customer_details_list.append(customer_details)
            logger.info(f"Extracted {len(customer_details_list)} customer profiles from CSV '{csv_path}'")
            return customer_details_list
        except Exception as e:
            logger.error(f"Failed to extract customer details from CSV '{csv_path}': {str(e)}")
            return []

    def get_text_embedding(self, text: str) -> Optional[List[float]]:
        if not self.azure_client:
            logger.warning("Azure OpenAI client not available. Skipping embedding generation.")
            return None

        try:
            response = self.azure_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            logger.info(f"Generated embedding with dimension {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return None

    def get_file_hash(self, file_path: str) -> str:
        try:
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            file_hash = sha256.hexdigest()
            logger.info(f"Generated hash for file '{file_path}': {file_hash}")
            return file_hash
        except Exception as e:
            logger.error(f"Failed to generate hash for file '{file_path}': {str(e)}")
            raise

    def list_all_documents(self, collection_name: str) -> List[Dict]:
        if not self.client.collections.exists(collection_name):
            logger.info(f"Collection '{collection_name}' does not exist")
            return []

        try:
            collection = self.client.collections.get(collection_name)
            results = collection.query.fetch_objects(
                return_properties=["title", "content", "category"]
            )

            documents = [
                {
                    "title": result.properties.get('title', ''),
                    "content": result.properties.get('content', ''),
                    "category": result.properties.get('category', ''),
                    "id": str(result.uuid)
                }
                for result in results.objects
            ]
            logger.info(f"Retrieved {len(documents)} documents from collection '{collection_name}'")
            return documents
        except Exception as e:
            logger.error(f"Failed to list documents from collection '{collection_name}': {str(e)}")
            return []

    def list_collections(self) -> List[Dict]:
        try:
            collections = self.client.collections.list_all(simple=False)
            collection_list = [
                {
                    "name": collection.name,
                    "description": collection.description,
                    "properties": [
                        {"name": prop.name, "data_type": str(prop.data_type), "description": prop.description}
                        for prop in collection.properties
                    ],
                    "vectorizer": collection.vectorizer.value if collection.vectorizer else "none"
                }
                for collection in collections.values()
            ]
            logger.info(f"Retrieved {len(collection_list)} collections from Weaviate")
            return collection_list
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            raise

    def migrate_collection(self, old_collection_name: str, new_collection_name: str) -> None:
        if not self.client.collections.exists(old_collection_name):
            logger.info(f"Old collection '{old_collection_name}' does not exist. Nothing to migrate.")
            return

        if not self.client.collections.exists(new_collection_name):
            logger.info(f"New collection '{new_collection_name}' does not exist. Creating it.")
            self.create_collection(new_collection_name, f"Migrated from {old_collection_name}")

        try:
            old_collection = self.client.collections.get(old_collection_name)
            new_collection = self.client.collections.get(new_collection_name)

            results = old_collection.query.fetch_objects(
                return_properties=["title", "content", "category"],
                limit=10000
            )

            for obj in results.objects:
                title = obj.properties.get("title", "Untitled")
                content = obj.properties.get("content", "")
                category = obj.properties.get("category", "")
                embedding = None
                if self.azure_client:
                    try:
                        embedding = self.get_text_embedding(content)
                    except Exception as e:
                        logger.warning(f"Embedding failed for '{title}': {str(e)}. Proceeding without embedding.")
                new_collection.data.insert(
                    properties={
                        "title": title,
                        "content": content,
                        "category": category
                    },
                    vector=embedding
                )
                logger.info(f"Migrated object: {title}")

            logger.info(f"Migration from '{old_collection_name}' to '{new_collection_name}' completed")
        except Exception as e:
            logger.error(f"Failed to migrate collection '{old_collection_name}' to '{new_collection_name}': {str(e)}")
            raise

    def delete_all_data(self) -> Dict:
        try:
            for collection_name in self.collection_names:
                if self.client.collections.exists(collection_name):
                    self.client.collections.delete(collection_name)
                    logger.info(f"Collection '{collection_name}' deleted")
            
            with self.lock:
                if os.path.exists(self.processed_files_path):
                    os.remove(self.processed_files_path)
                    logger.info(f"Cleared processed files tracker at {self.processed_files_path}")
            
            self.create_collection("CustomerProfiles", "Collection for customer profile data")
            self.create_collection("InsuranceDocs", "Collection for insurance document data")
            
            return {
                "status": "success",
                "message": "All data in 'CustomerProfiles' and 'InsuranceDocs' deleted and collections recreated"
            }
        except Exception as e:
            logger.error(f"Failed to delete data in collections: {str(e)}")
            raise

    def upload_files_from_folder(self, folder_path: str, collection_name: str) -> List[Dict]:
        if not os.path.isdir(folder_path):
            raise ValueError(f"Folder not found: {folder_path}")

        results = []
        try:
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                try:
                    if file_name.lower().endswith('.csv') and collection_name == "CustomerProfiles":
                        result = self.add_customer_profile_csv(collection_name, file_path)
                        results.append(result)
                    elif file_name.lower().endswith('.pdf') and collection_name == "InsuranceDocs":
                        result = self.add_insurance_document_pdf(collection_name, file_path)
                        results.append(result)
                    else:
                        logger.info(f"Skipping file: {file_name} (unsupported format for collection '{collection_name}')")
                except Exception as e:
                    error_result = {
                        "title": file_name,
                        "status": "error",
                        "message": str(e)
                    }
                    results.append(error_result)
                    logger.error(f"Failed to upload file '{file_path}': {str(e)}")
        except Exception as e:
            logger.error(f"Failed to process folder '{folder_path}': {str(e)}")
            raise
        return results

    def close(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()
            logger.info("Weaviate client connection closed properly")