import os
import re
import json
import logging
import asyncio
import io
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from openai import AzureOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from gtts import gTTS
import speech_recognition as sr
import weaviate
import weaviate.classes as wvc
from weaviate.auth import AuthApiKey
from langdetect import detect, DetectorFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("insurance_assistant.log")]
)
logger = logging.getLogger(__name__)

# Configuration
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
    "url": "nwhexgobshomixnhqbhlg.c0.asia-southeast1.gcp.weaviate.cloud",
    "api_key": "STV3MHJ1ZEl0OTJaTDJaY182VXRuSXVTUlQ1UWxDKzFTRGgrNlFsdlpnMzlVeGxWMkYwemFDdHh0dC9FPV92MjAw",
    "class_name": "singaporeclusters",
    "collection_names": ["CustomerProfiles", "InsuranceDocs"]
}

LANGUAGES = {
    "english": {"code": "en", "display": "English", "google_code": "en-US"},
    "malay": {"code": "ms", "display": "Malay", "google_code": "ms-MY"},
    "thai": {"code": "th", "display": "Thai", "google_code": "th-TH"},
    "chinese": {"code": "zh-CN", "display": "Chinese", "google_code": "zh-CN"},
    "tamil": {"code": "ta", "display": "Tamil", "google_code": "ta-IN"},
    "telugu": {"code": "te", "display": "Telugu", "google_code": "te-IN"},
    "singlish": {"code": "en", "display": "Singlish", "google_code": "en-SG"}
}

# Ensure consistent language detection results
DetectorFactory.seed = 0

class WeaviateManager:
    def __init__(self, config: Dict, azure_config: Dict):
        self.client = None
        self.collection_names = config["collection_names"]
        self.azure_client = None
        self.memory = ChatMessageHistory()
        self.embedding_model = azure_config["embedding_deployment"]

        try:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=WEAVIATE_CONFIG["url"],
                auth_credentials=AuthApiKey(WEAVIATE_CONFIG["api_key"]),
            )
            for collection_name in self.collection_names:
                if not self.client.collections.exists(collection_name):
                    logger.error(f"Collection '{collection_name}' does not exist.")
                    raise ValueError(f"Collection '{collection_name}' not found in Weaviate.")
            logger.info(f"Connected to Weaviate. Using collections: {', '.join(self.collection_names)}.")
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {e}")
            raise

        try:
            self.azure_client = AzureOpenAI(
                azure_endpoint=azure_config["endpoint"],
                api_key=azure_config["api_key"],
                api_version=azure_config["api_version"]
            )
            logger.info("Connected to Azure OpenAI service.")
        except Exception as e:
            logger.error(f"Error initializing Azure OpenAI client: {e}")
            raise

    def __del__(self):
        if self.client:
            try:
                self.client.close()
                logger.info("Weaviate client connection closed.")
            except Exception as e:
                logger.error(f"Error closing Weaviate client: {e}")

    async def get_text_embedding(self, text: str) -> Optional[List[float]]:
        if not self.azure_client:
            logger.error("Azure OpenAI client not initialized.")
            return None
        try:
            max_length = 8000
            if len(text) > max_length:
                text = text[:max_length]
                logger.warning("Text truncated to %d characters for embedding", max_length)
            response = await asyncio.to_thread(
                self.azure_client.embeddings.create,
                input=text,
                model=self.embedding_model
            )
            embedding = response.data[0].embedding
            if len(embedding) != 1536:
                logger.error(f"Expected 1536-dimensional embedding, got {len(embedding)}")
                return None
            return embedding
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return None

    def classify_query_intent(self, query_text: str) -> str:
        customer_patterns = [
            r'\bmy\s+(?:profile|policy|details|information)\b',
            r'\bcustomer\s+(?:profile|details|information)\b',
            r'\bwho\s+(?:am\s+i|is\s+this\s+customer)\b',
            r'\bmy\s+(?:current|existing)\s+policy\b',
            r'\bprofile\s+(?:summary|details)\b',
            r'\bcustomer\s+id\b',
            r'\bmy\s+(?:age|occupation|recommendations)\b'
        ]
        product_patterns = [
            r'\b(?:what|which|tell\s+me\s+about)\s+(?:insurance|plans?|policies)\b',
            r'\blist\s+(?:all\s+)?(?:insurance|products|plans)\b',
            r'\b(?:benefits|coverage|features)\s+of\b',
            r'\b(?:premium|cost|price)\s+(?:of|for)\b',
            r'\b(?:critical\s+illness|life\s+insurance|health\s+insurance)\b',
            r'\b(?:recommend|suggest)\s+(?:insurance|policy)\b'
        ]
        recommendation_patterns = [
            r'\brecommend\s+(?:insurance|policy|plan)\b',
            r'\bbest\s+(?:insurance|policy|plan)\s+for\s+me\b',
            r'\bwhat\s+(?:insurance|policy)\s+should\s+i\b',
            r'\bsuitable\s+(?:insurance|policy)\b'
        ]
        query_lower = query_text.lower()
        if any(re.search(pattern, query_lower) for pattern in recommendation_patterns):
            return "recommendation"
        elif any(re.search(pattern, query_lower) for pattern in customer_patterns):
            return "customer_specific"
        elif any(re.search(pattern, query_lower) for pattern in product_patterns):
            return "product_general"
        else:
            return "general"

    async def check_customer_name_ambiguity(self, customer_name: str) -> List[str]:
        if not self.client or not customer_name:
            return []
        try:
            collection = self.client.collections.get("CustomerProfiles")
            result = await asyncio.to_thread(
                collection.query.bm25,
                query=customer_name,
                limit=10,
                return_properties=["title", "content"]
            )
            customer_ids = []
            for obj in result.objects:
                try:
                    content = json.loads(obj.properties.get('content', '{}'))
                    stored_name = content.get('Customer Name', '').lower()
                    if customer_name.lower() in stored_name or stored_name in customer_name.lower():
                        customer_ids.append(obj.properties.get('title', ''))
                except:
                    continue
            return customer_ids[:5]
        except Exception as e:
            logger.error(f"Error checking customer name ambiguity: {e}")
            return []

    async def get_customer_summary(self, customer_id: str) -> str:
        if not self.client or not customer_id:
            return ""
        try:
            collection = self.client.collections.get("CustomerProfiles")
            where_filter = wvc.query.Filter.by_property("title").equal(customer_id)
            result = await asyncio.to_thread(
                collection.query.fetch_objects,
                where=where_filter,
                limit=1,
                return_properties=["title", "content", "category"]
            )
            if not result.objects:
                return ""
            customer_data = result.objects[0].properties.get('content', '{}')
            try:
                parsed_data = json.loads(customer_data)
                # Dynamically construct summary from all available fields
                summary_lines = []
                for key, value in parsed_data.items():
                    if isinstance(value, dict):
                        # Handle nested dictionaries (e.g., current_policy)
                        nested_info = ", ".join(f"{k}: {v}" for k, v in value.items() if v)
                        if nested_info:
                            summary_lines.append(f"{key.replace('_', ' ').title()}: {nested_info}")
                    elif isinstance(value, list):
                        # Handle lists (e.g., recommendations)
                        if value:
                            summary_lines.append(f"{key.replace('_', ' ').title()}: {', '.join(str(v) for v in value)}")
                    elif value:
                        summary_lines.append(f"{key.replace('_', ' ').title()}: {value}")
                summary = "Customer Profile Summary: " + "; ".join(summary_lines) if summary_lines else "No details available."
                return summary
            except json.JSONDecodeError:
                return customer_data[:500]
        except Exception as e:
            logger.error(f"Error getting customer summary: {e}")
            return ""

    async def search_query_in_collections(self, query_text: str, original_query: str, top_k: int = 5, customer_id: Optional[str] = None) -> str:
        if not query_text.strip():
            logger.warning("Empty query received.")
            return "I didn't receive any query. How can I help you with insurance?"
        if not self.client:
            logger.error("Weaviate client not initialized.")
            return "I'm sorry, I cannot access the insurance data. Please try again later."
        try:
            query_intent = self.classify_query_intent(original_query)
            logger.info(f"Query intent classified as: {query_intent}")
            if not customer_id and query_intent in ["customer_specific", "recommendation"]:
                name_match = re.search(r'\bfor\s+([A-Za-z\s]+)\b|\bmy\s+name\s+is\s+([A-Za-z\s]+)\b', original_query, re.IGNORECASE)
                if name_match:
                    customer_name = (name_match.group(1) or name_match.group(2)).strip()
                    potential_ids = await self.check_customer_name_ambiguity(customer_name)
                    if len(potential_ids) > 1:
                        return f"I found multiple customers with similar names. Please provide your customer ID to get accurate information. Available customer IDs: {', '.join(potential_ids)}"
            collections_to_search = []
            if query_intent == "customer_specific" and customer_id:
                collections_to_search = ["CustomerProfiles"]
            elif query_intent == "product_general":
                collections_to_search = ["InsuranceDocs"]
            elif query_intent == "recommendation":
                collections_to_search = ["CustomerProfiles", "InsuranceDocs"] if customer_id else ["InsuranceDocs"]
            else:
                collections_to_search = ["InsuranceDocs"]
            if query_intent == "customer_specific" and customer_id and any(word in original_query.lower() for word in ["summary", "profile", "details"]):
                customer_summary = await self.get_customer_summary(customer_id)
                if customer_summary:
                    self.memory.add_user_message(original_query)
                    response = f"Here's your complete profile information: {customer_summary}"
                    self.memory.add_ai_message(response)
                    return response
            query_embedding = await self.get_text_embedding(query_text)
            if not query_embedding:
                logger.error("Failed to generate query embedding.")
                return "I'm sorry, I couldn't process your query. Please try again."
            results = []
            for collection_name in collections_to_search:
                collection = self.client.collections.get(collection_name)
                where_filter = None
                if customer_id and collection_name == "CustomerProfiles":
                    where_filter = wvc.query.Filter.by_property("title").equal(customer_id)
                result = await asyncio.to_thread(
                    collection.query.near_vector,
                    near_vector=query_embedding,
                    limit=top_k,
                    distance=0.8,
                    return_properties=["title", "content", "category"],
                    return_metadata=wvc.query.MetadataQuery(distance=True),
                    filters=where_filter
                )
                results.extend(result.objects)
                logger.info(f"Found {len(result.objects)} results in {collection_name} for query: {query_text}")
            results = sorted(results, key=lambda x: x.metadata.distance if x.metadata else float('inf'))[:top_k]
            self.memory.add_user_message(original_query)
            if not results:
                logger.info("No relevant results found in Weaviate.")
                return f"I couldn't find any information matching your query in our insurance database{' for this customer' if customer_id and query_intent == 'customer_specific' else ''}. Can you provide more details or try a different question?"
            else:
                response = await self.format_results_with_llm(results, original_query, query_text, customer_id, query_intent)
                logger.info(f"Generated response: {response}")
            self.memory.add_ai_message(response)
            return response
        except Exception as e:
            logger.error(f"Error in search process: {e}")
            return "I'm sorry, an error occurred while processing your query. Please try again."

    async def format_results_with_llm(self, details: List, original_query: str, enhanced_query: str, customer_id: Optional[str] = None, query_intent: str = "general") -> str:
        if not self.azure_client:
            logger.warning("Azure client not available, returning raw content.")
            return "Based on your query, I found some information: " + " ".join([d.properties.get('content', 'No content available')[:200] for d in details[:2]])
        try:
            details_text = ""
            for i, detail in enumerate(details[:5]):
                title = detail.properties.get('title', 'Unknown')
                content = detail.properties.get('content', 'No content available')
                category = detail.properties.get('category', 'Unknown')
                if category == "customer_profile":
                    try:
                        parsed_details = json.loads(content)
                        content = json.dumps(parsed_details, indent=2)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON details for title {title}")
                details_text += f"Document {i+1}: Category: {category}, Title: {title}, Details: {content}\n"
            chat_history = self.memory.messages
            if query_intent == "customer_specific":
                instruction = (
                    "You are an AI assistant for Major Hong Leong Assurance in Singapore. "
                    "The user is asking for customer-specific information. Provide a comprehensive summary of the customer's profile including all available details. "
                    "Present the information in a clear, organized manner without using markup or bullet points. "
                    "Query Intent: {query_intent} "
                    "User Query: {original_query} "
                    "Customer ID: {customer_id} "
                    "Retrieved Data: {details} "
                    "Provide complete customer information in a conversational format."
                )
            elif query_intent == "product_general":
                instruction = (
                    "You are an AI assistant for Major Hong Leong Assurance in Singapore. "
                    "The user is asking about insurance products in general. Focus only on product information from InsuranceDocs. "
                    "Do not include any customer-specific information or names. "
                    "Query Intent: {query_intent} "
                    "User Query: {original_query} "
                    "Retrieved Data: {details} "
                    "Provide accurate product information in a conversational format without markup."
                )
            else:
                instruction = (
                    "You are an AI assistant for Major Hong Leong Assurance in Singapore, providing accurate information about insurance products and customer profiles based solely on the provided data. "
                    "Query Intent: {query_intent} "
                    "User Query: {original_query} "
                    "Enhanced Query: {enhanced_query} "
                    "Customer ID: {customer_id} "
                    "Retrieved Data: {details} "
                    "Provide a helpful response using only the retrieved information in a conversational format without markup."
                )
            instruction += (
                " Follow these rules: "
                "Write in a conversational, human-readable style. "
                "Do NOT use markup language or symbols. "
                "Do NOT use bullet points, numbered lists, or headings. "
                "Keep responses concise and informative. "
                "Maintain a friendly and professional tone."
            )
            prompt_template = PromptTemplate(
                input_variables=["query_intent", "original_query", "enhanced_query", "details", "customer_id"],
                template=instruction
            )
            refined_prompt = prompt_template.format(
                query_intent=query_intent,
                original_query=original_query,
                enhanced_query=enhanced_query,
                details=details_text,
                customer_id=customer_id or "None"
            )
            response = await asyncio.to_thread(
                self.azure_client.chat.completions.create,
                model=AZURE_CONFIG["gpt_deployment"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI assistant for Major Hong Leong Assurance in Singapore. "
                            "Provide accurate information based only on the provided data. "
                            "Always respond in a conversational style without markup language. "
                            "Keep responses concise and professional."
                        )
                    },
                    {"role": "user", "content": refined_prompt}
                ],
                temperature=0.5,
                max_tokens=800
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return "I found some information, but I couldn't process it properly. Can you rephrase your query?"

    async def detect_language(self, text: str) -> str:
        try:
            detected_lang = detect(text)
            lang_map = {
                'en': 'english',
                'ms': 'malay',
                'th': 'thai',
                'zh-cn': 'chinese',
                'ta': 'tamil',
                'te': 'telugu',
                'en-sg': 'singlish'
            }
            for code, lang in lang_map.items():
                if detected_lang.startswith(code):
                    return lang
            return 'english'
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return 'english'

    async def translate_text(self, text: str, src_lang: str, dest_lang: str) -> str:
        if not text.strip() or src_lang == dest_lang or not self.azure_client:
            return text
        try:
            lang_map = {lang_info["code"]: lang_info["display"] for _, lang_info in LANGUAGES.items()}
            src_lang_name = lang_map.get(src_lang, "English")
            dest_lang_name = lang_map.get(dest_lang, "English")
            response = await asyncio.to_thread(
                self.azure_client.chat.completions.create,
                model=AZURE_CONFIG["gpt_deployment"],
                messages=[
                    {
                        "role": "system",
                        "content": "Translate accurately. Return only the translated text without markup or explanations."
                    },
                    {"role": "user", "content": f"Translate from {src_lang_name} to {dest_lang_name}: '{text}'"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            translated_text = response.choices[0].message.content.strip()
            logger.info(f"Translated from {src_lang_name} to {dest_lang_name}")
            return translated_text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

class MultilingualVoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 500
        self.weaviate_manager = WeaviateManager(WEAVIATE_CONFIG, AZURE_CONFIG)
        self.user_language = "en"
        self.display_language = "english"
        self.input_mode = "text"
        self.output_mode = "text"
        self.conversation_history = {}
        self.load_conversation_history()
        logger.info("Multilingual Voice Assistant initialized")

    def load_conversation_history(self):
        try:
            if os.path.exists('conversation_history.json'):
                with open('conversation_history.json', 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                logger.info("Loaded conversation history from file")
            else:
                logger.info("No saved conversation history found")
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")
            self.conversation_history = {}

    def save_conversation(self, filename='conversation_history.json'):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            logger.info(f"Conversation saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False

    async def translate_to_english(self, text: str, src_lang: str) -> str:
        if src_lang == "en":
            return text
        try:
            translated_text = await self.weaviate_manager.translate_text(text, src_lang, "en")
            logger.info(f"Translated query to English: {translated_text}")
            return translated_text
        except Exception as e:
            logger.error(f"Error translating to English: {e}")
            return text

    async def enhance_query(self, query: str, query_intent: str) -> str:
        if not self.weaviate_manager.azure_client:
            return query
        try:
            if query_intent == "customer_specific":
                prompt = f"Optimize this customer profile query for database search: '{query}'"
            elif query_intent == "product_general":
                prompt = f"Optimize this insurance product query for database search: '{query}'"
            else:
                prompt = f"Optimize this insurance query for database search: '{query}'"
            response = await asyncio.to_thread(
                self.weaviate_manager.azure_client.chat.completions.create,
                model=AZURE_CONFIG["gpt_deployment"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100
            )
            enhanced_query = response.choices[0].message.content.strip()
            logger.info(f"Enhanced query: {enhanced_query}")
            return enhanced_query
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query

    async def process_user_input(self, text_input: str, session_id: str, language: str = "english", input_mode: str = "text", output_mode: str = "text", customer_id: Optional[str] = None) -> Tuple[str, Optional[bytes], List[str]]:
        try:
            self.set_language(language)
            self.set_input_mode(input_mode)
            self.set_output_mode(output_mode)
            if self.input_mode == "voice":
                detected_lang = await self.weaviate_manager.detect_language(text_input)
                self.set_language(detected_lang)
                logger.info(f"Detected language for voice input: {detected_lang}")
            original_query = text_input
            query_intent = self.weaviate_manager.classify_query_intent(original_query)
            if self.display_language != "english":
                text_input = await self.translate_to_english(text_input, self.user_language)
            enhanced_query = await self.enhance_query(text_input, query_intent)
            entities = self.extract_entities(original_query)
            is_recommendation = self.is_recommend_insurance_intent(original_query)
            response = await self.generate_response(enhanced_query, session_id, entities, is_recommendation, original_query, customer_id, query_intent)
            if self.display_language != "english":
                response = await self.weaviate_manager.translate_text(response, "en", self.user_language)
            suggestions = await self.generate_suggestions(original_query, response, session_id, query_intent, is_recommendation, customer_id)
            audio_data = None
            if self.output_mode in ["voice", "both"]:
                audio_data = await self.generate_audio(response)
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = {"messages": [], "summary": "", "entities": {}, "pending_intent": None, "customer_id": customer_id}
            timestamp = datetime.now().isoformat()
            self.conversation_history[session_id]["messages"].append({
                "role": "user",
                "content": original_query,
                "timestamp": timestamp,
                "language": self.display_language,
                "intent": query_intent
            })
            self.conversation_history[session_id]["messages"].append({
                "role": "assistant",
                "content": response,
                "timestamp": timestamp,
                "language": self.display_language,
                "suggestions": suggestions
            })
            self.conversation_history[session_id]["entities"].update(entities)
            self.conversation_history[session_id]["pending_intent"] = "recommendation" if is_recommendation else None
            self.conversation_history[session_id]["customer_id"] = customer_id
            self.save_conversation()
            logger.info(f"Processed input with intent {query_intent}: {original_query[:50]}...")
            return response, audio_data, suggestions
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            raise

    async def generate_response(self, enhanced_query: str, session_id: str, entities: Dict, is_recommendation: bool, original_query: str, customer_id: Optional[str] = None, query_intent: str = "general") -> str:
        try:
            if not self.weaviate_manager.azure_client:
                logger.error("Azure OpenAI client not initialized.")
                return "Sorry, I couldn't process your request due to a configuration issue."
            search_query = enhanced_query
            if is_recommendation and entities:
                entity_info = []
                if entities["age"]:
                    entity_info.append(f"age {entities['age']}")
                if entities["gender"]:
                    entity_info.append(f"{entities['gender']}")
                if entities["location"]:
                    entity_info.append(f"residing in {entities['location']}")
                if entity_info:
                    search_query += f" (for someone {', '.join(entity_info)})"
            response = await self.weaviate_manager.search_query_in_collections(
                search_query,
                original_query=original_query,
                customer_id=customer_id
            )
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I couldn't process your request."

    async def generate_suggestions(self, user_input: str, response: str, session_id: str, query_intent: str = "general", is_recommendation: bool = False, customer_id: Optional[str] = None) -> List[str]:
        try:
            if not self.weaviate_manager.azure_client:
                return self.get_default_suggestions(query_intent, is_recommendation)
            if query_intent == "recommendation" or is_recommendation:
                return await self.generate_recommendation_suggestions(user_input, response, customer_id, session_id)
            elif query_intent == "customer_specific":
                return await self.generate_customer_suggestions(user_input, response, customer_id, session_id)
            else:
                return await self.generate_general_suggestions(user_input, response, session_id, customer_id)
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return self.get_default_suggestions(query_intent, is_recommendation)

    async def generate_recommendation_suggestions(self, user_input: str, response: str, customer_id: Optional[str], session_id: str) -> List[str]:
        try:
            customer_data = ""
            if customer_id:
                customer_data = await self.weaviate_manager.get_customer_summary(customer_id)
            collection = self.weaviate_manager.client.collections.get("InsuranceDocs")
            result = await asyncio.to_thread(
                collection.query.fetch_objects,
                limit=5,
                return_properties=["title", "content", "category"]
            )
            insurance_data = ""
            for i, obj in enumerate(result.objects):
                insurance_data += f"Document {i+1}: {obj.properties.get('content', 'No content')}\n"
            session = self.conversation_history.get(session_id, {})
            messages = session.get("messages", [])[-4:]
            context = json.dumps(messages[-2:], indent=1) if messages else "No recent conversation."
            prompt = f"""
            Based on the user query, response, customer data, and insurance documents, generate:
            1. One relevant follow-up question for an insurance recommendation conversation.
            2. Two insurance product recommendations in the format 'Product Name: One-line description.'
            Use customer data (if available) to tailor the recommendations based on available attributes.
            Return a JSON object with keys "question" and "recommendations", where "question" is a string and "recommendations" is an array of two strings.
            Customer Data: {customer_data}
            Insurance Documents: {insurance_data}
            Recent Conversation: {context}
            User Query: {user_input}
            Response: {response}
            Example output:
            {{
                "question": "What are the premium payment options for this plan?",
                "recommendations": [
                    "SMART Saver Plus: A savings plan with flexible premiums.",
                    "HealthShield Gold: Comprehensive medical coverage."
                ]
            }}
            """
            completion = await asyncio.to_thread(
                self.weaviate_manager.azure_client.chat.completions.create,
                model=AZURE_CONFIG["gpt_deployment"],
                messages=[
                    {
                        "role": "system",
                        "content": "Generate one follow-up question and two insurance product recommendations based on the provided information. Return a JSON object with keys 'question' and 'recommendations'."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            raw_response = completion.choices[0].message.content.strip()
            try:
                result = json.loads(raw_response)
                question = result.get("question", "What are the premium payment options for this plan?")
                recommendations = result.get("recommendations", [
                    "SMART Saver Plus: A savings plan with flexible premiums.",
                    "HealthShield Gold: Comprehensive medical coverage."
                ])
                return [question] + recommendations
            except json.JSONDecodeError:
                return [
                    "What are the premium payment options for this plan?",
                    "SMART Saver Plus: A savings plan with flexible premiums.",
                    "HealthShield Gold: Comprehensive medical coverage."
                ]
        except Exception as e:
            logger.error(f"Error generating recommendation suggestions: {e}")
            return [
                "What are the premium payment options for this plan?",
                "SMART Saver Plus: A savings plan with flexible premiums.",
                "HealthShield Gold: Comprehensive medical coverage."
            ]

    async def generate_customer_suggestions(self, user_input: str, response: str, customer_id: Optional[str], session_id: str) -> List[str]:
        if not customer_id:
            return [
                "Can you provide your customer ID for personalized information?",
                "SMART Saver Plus: A savings plan with flexible premiums.",
                "HealthShield Gold: Comprehensive medical coverage."
            ]
        try:
            customer_data = await self.weaviate_manager.get_customer_summary(customer_id)
            collection = self.weaviate_manager.client.collections.get("InsuranceDocs")
            result = await asyncio.to_thread(
                collection.query.fetch_objects,
                limit=5,
                return_properties=["title", "content", "category"]
            )
            insurance_data = ""
            for i, obj in enumerate(result.objects):
                insurance_data += f"Document {i+1}: {obj.properties.get('content', 'No content')}\n"
            session = self.conversation_history.get(session_id, {})
            messages = session.get("messages", [])[-4:]
            context = json.dumps(messages[-2:], indent=1) if messages else "No recent conversation."
            prompt = f"""
            Based on the user query, response, customer data, and insurance documents, generate:
            1. One relevant follow-up question for a customer-specific insurance conversation.
            2. Two insurance product recommendations in the format 'Product Name: One-line description.'
            Use customer data to tailor the recommendations based on available attributes.
            Return a JSON object with keys "question" and "recommendations", where "question" is a string and "recommendations" is an array of two strings.
            Customer Data: {customer_data}
            Insurance Documents: {insurance_data}
            Recent Conversation: {context}
            User Query: {user_input}
            Response: {response}
            Example output:
            {{
                "question": "Would you like to update any of your profile information?",
                "recommendations": [
                    "SMART Saver Plus: A savings plan with flexible premiums.",
                    "HealthShield Gold: Comprehensive medical coverage."
                ]
            }}
            """
            completion = await asyncio.to_thread(
                self.weaviate_manager.azure_client.chat.completions.create,
                model=AZURE_CONFIG["gpt_deployment"],
                messages=[
                    {
                        "role": "system",
                        "content": "Generate one follow-up question and two insurance product recommendations based on the provided information. Return a JSON object with keys 'question' and 'recommendations'."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            raw_response = completion.choices[0].message.content.strip()
            try:
                result = json.loads(raw_response)
                question = result.get("question", "Would you like to update any of your profile information?")
                recommendations = result.get("recommendations", [
                    "SMART Saver Plus: A savings plan with flexible premiums.",
                    "HealthShield Gold: Comprehensive medical coverage."
                ])
                return [question] + recommendations
            except json.JSONDecodeError:
                return [
                    "Would you like to update any of your profile information?",
                    "SMART Saver Plus: A savings plan with flexible premiums.",
                    "HealthShield Gold: Comprehensive medical coverage."
                ]
        except Exception as e:
            logger.error(f"Error generating customer suggestions: {e}")
            return [
                "Would you like to update any of your profile information?",
                "SMART Saver Plus: A savings plan with flexible premiums.",
                "HealthShield Gold: Comprehensive medical coverage."
            ]

    async def generate_general_suggestions(self, user_input: str, response: str, session_id: str, customer_id: Optional[str]) -> List[str]:
        try:
            customer_data = ""
            if customer_id:
                customer_data = await self.weaviate_manager.get_customer_summary(customer_id)
            collection = self.weaviate_manager.client.collections.get("InsuranceDocs")
            result = await asyncio.to_thread(
                collection.query.fetch_objects,
                limit=5,
                return_properties=["title", "content", "category"]
            )
            insurance_data = ""
            for i, obj in enumerate(result.objects):
                insurance_data += f"Document {i+1}: {obj.properties.get('content', 'No content')}\n"
            session = self.conversation_history.get(session_id, {})
            messages = session.get("messages", [])[-4:]
            context = json.dumps(messages[-2:], indent=1) if messages else "No recent conversation."
            prompt = f"""
            Based on the user query, response, customer data, and insurance documents, generate:
            1. One relevant follow-up question for a general insurance conversation.
            2. Two insurance product recommendations in the format 'Product Name: One-line description.'
            Use customer data (if available) to tailor the recommendations based on available attributes.
            Return a JSON object with keys "question" and "recommendations", where "question" is a string and "recommendations" is an array of two strings.
            Customer Data: {customer_data}
            Insurance Documents: {insurance_data}
            Recent Conversation: {context}
            User Query: {user_input}
            Response: {response}
            Example output:
            {{
                "question": "How do I apply for this insurance plan?",
                "recommendations": [
                    "SMART Saver Plus: A savings plan with flexible premiums.",
                    "HealthShield Gold: Comprehensive medical coverage."
                ]
            }}
            """
            completion = await asyncio.to_thread(
                self.weaviate_manager.azure_client.chat.completions.create,
                model=AZURE_CONFIG["gpt_deployment"],
                messages=[
                    {
                        "role": "system",
                        "content": "Generate one follow-up question and two insurance product recommendations based on the provided information. Return a JSON object with keys 'question' and 'recommendations'."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            raw_response = completion.choices[0].message.content.strip()
            try:
                result = json.loads(raw_response)
                question = result.get("question", "How do I apply for this insurance plan?")
                recommendations = result.get("recommendations", [
                    "SMART Saver Plus: A savings plan with flexible premiums.",
                    "HealthShield Gold: Comprehensive medical coverage."
                ])
                return [question] + recommendations
            except json.JSONDecodeError:
                return [
                    "How do I apply for this insurance plan?",
                    "SMART Saver Plus: A savings plan with flexible premiums.",
                    "HealthShield Gold: Comprehensive medical coverage."
                ]
        except Exception as e:
            logger.error(f"Error generating general suggestions: {e}")
            return [
                "How do I apply for this insurance plan?",
                "SMART Saver Plus: A savings plan with flexible premiums.",
                "HealthShield Gold: Comprehensive medical coverage."
            ]

    def get_default_suggestions(self, query_intent: str, is_recommendation: bool) -> List[str]:
        if is_recommendation or query_intent == "recommendation":
            return [
                "What are the premium payment options for this plan?",
                "SMART Saver Plus: A savings plan with flexible premiums.",
                "HealthShield Gold: Comprehensive medical coverage."
            ]
        elif query_intent == "customer_specific":
            return [
                "Would you like to update any of your profile information?",
                "SMART Saver Plus: A savings plan with flexible premiums.",
                "HealthShield Gold: Comprehensive medical coverage."
            ]
        elif query_intent == "product_general":
            return [
                "How do I apply for this insurance plan?",
                "SMART Saver Plus: A savings plan with flexible premiums.",
                "HealthShield Gold: Comprehensive medical coverage."
            ]
        else:
            return [
                "Can you tell me more about available insurance plans?",
                "SMART Saver Plus: A savings plan with flexible premiums.",
                "HealthShield Gold: Comprehensive medical coverage."
            ]

    async def generate_session_summary(self, session_id: str) -> str:
        try:
            if not self.weaviate_manager.azure_client:
                return "Summary unavailable."
            messages = self.conversation_history.get(session_id, {}).get("messages", [])
            if not messages:
                return "No conversation yet."
            if len(messages) < 4:
                return "Initial conversation in progress."
            prompt = f"""
            Summarize this insurance conversation in 1-2 sentences focusing on key topics discussed:
            {json.dumps(messages[-6:], indent=1)}
            """
            completion = await asyncio.to_thread(
                self.weaviate_manager.azure_client.chat.completions.create,
                model=AZURE_CONFIG["gpt_deployment"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.5
            )
            summary = completion.choices[0].message.content.strip()
            logger.info(f"Generated summary for session {session_id}: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return "Summary unavailable."

    async def generate_audio(self, text: str) -> Optional[bytes]:
        try:
            lang_code = LANGUAGES.get(self.display_language, {"code": "en"})["code"]
            audio_bytes = self.text_to_speech(text, lang_code)
            if audio_bytes:
                logger.info(f"Generated audio for text: {text[:50]}...")
            else:
                logger.warning("Failed to generate audio.")
            return audio_bytes
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return None

    def text_to_speech(self, text: str, lang_code: str) -> Optional[bytes]:
        if not text or not text.strip():
            logger.warning("Empty text for TTS, skipping")
            return None
        try:
            supported_langs = ["en", "ms", "th", "zh-CN", "ta", "te"]
            lang_code = lang_code if lang_code in supported_langs else "en"
            tts = gTTS(text=text, lang=lang_code, slow=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            audio_bytes = mp3_fp.read()
            logger.info(f"Generated TTS audio in memory for language: {lang_code}")
            return audio_bytes
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None

    def speech_to_text(self, timeout: int = 60) -> Optional[str]:
        with sr.Microphone() as source:
            logger.info("Listening for speech input...")
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=timeout)
                logger.info("Processing speech...")
                google_lang_code = LANGUAGES.get(self.display_language, {"google_code": "en-US"})["google_code"]
                text = self.recognizer.recognize_google(audio, language=google_lang_code)
                logger.info(f"Recognized speech: {text}")
                return text
            except sr.WaitTimeoutError:
                logger.warning("No speech detected within timeout period")
                return None
            except sr.UnknownValueError:
                logger.warning("Could not understand audio")
                return None
            except sr.RequestError as e:
                logger.error(f"Could not request results from Google Speech Recognition service; {e}")
                return None
            except Exception as e:
                logger.error(f"Error in speech recognition: {e}")
                return None

    def set_input_mode(self, mode: str) -> bool:
        if mode.lower() in ["text", "voice"]:
            self.input_mode = mode.lower()
            logger.info(f"Input mode set to {self.input_mode}")
            return True
        logger.warning(f"Invalid input mode: {mode}. Use 'text' or 'voice'.")
        return False

    def set_output_mode(self, mode: str) -> bool:
        if mode.lower() in ["text", "voice", "both"]:
            self.output_mode = mode.lower()
            logger.info(f"Output mode set to {self.output_mode}")
            return True
        logger.warning(f"Invalid output mode: {mode}. Use 'text', 'voice', or 'both'.")
        return False

    def set_language(self, language: str) -> bool:
        language = language.lower()
        if language in LANGUAGES:
            self.user_language = LANGUAGES[language]["code"]
            self.display_language = language
            logger.info(f"Language set to {language} (code: {self.user_language})")
            return True
        logger.warning(f"Unsupported language: {language}. Using English.")
        self.user_language = "en"
        self.display_language = "english"
        return False

    def get_conversation_history(self):
        return self.conversation_history

    def extract_entities(self, message: str) -> Dict[str, any]:
        entities = {"age": None, "gender": None, "location": None}
        age_match = re.search(r'(\d{1,3})\s*(?:year|yr|years)?\s*(?:old)?', message, re.IGNORECASE)
        if age_match:
            entities["age"] = int(age_match.group(1))
        gender_match = re.search(r'\b(male|female|man|woman|gentleman|lady)\b', message, re.IGNORECASE)
        if gender_match:
            entities["gender"] = gender_match.group(1).lower()
            if entities["gender"] in ["man", "gentleman"]:
                entities["gender"] = "male"
            elif entities["gender"] in ["woman", "lady"]:
                entities["gender"] = "female"
        location_match = re.search(r'\b(?:in|residing in|from|live in)\s*([A-Za-z\s]+)\b', message, re.IGNORECASE)
        if location_match:
            entities["location"] = location_match.group(1).strip()
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def is_recommend_insurance_intent(self, message: str) -> bool:
        patterns = [
            r'insurance.*should\s+I',
            r'recommend.*insurance',
            r'what.*insurance.*for\s+me',
            r'best.*insurance',
            r'suitable.*insurance',
            r'suggest.*insurance'
        ]
        return any(re.search(pattern, message.lower()) for pattern in patterns)

    def clear_conversation_history(self) -> None:
        self.conversation_history = {}
        self.weaviate_manager.memory = ChatMessageHistory()
        logger.info("Conversation history cleared")