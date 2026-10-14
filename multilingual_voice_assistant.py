import os
import time
import speech_recognition as sr
from gtts import gTTS
import io
from openai import AzureOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
import asyncio
from functools import lru_cache
import logging
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from weaviate.auth import AuthApiKey
import weaviate
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("insurance_assistant.log")
    ]
)
logger = logging.getLogger(__name__)

# Language configuration with ISO codes and display names
LANGUAGES = {
    "english": {"code": "en", "display": "English", "google_code": "en-US"},
    "malay": {"code": "ms", "display": "Malay", "google_code": "ms-MY"},
    "thai": {"code": "th", "display": "Thai", "google_code": "th-TH"},
    "chinese": {"code": "zh-CN", "display": "Chinese", "google_code": "zh-CN"},
    "tamil": {"code": "ta", "display": "Tamil", "google_code": "ta-IN"},
    "telugu": {"code": "te", "display": "Telugu", "google_code": "te-IN"},
    "singlish": {"code": "en", "display": "Singlish", "google_code": "en-SG"}
}

class WeaviateManager:
    def __init__(self, config: Dict[str, Any], azure_config: Optional[Dict[str, str]] = None):
        """
        Initialize the WeaviateManager with connection to vector database and Azure OpenAI.
        """
        self.client = None
        self.collection_name = config["class_name"]
        self.azure_client = None
        self.memory = ChatMessageHistory()
        self.embedding_model = None
        
        try:
            self.client = weaviate.connect_to_wcs(
                cluster_url=config["url"],
                auth_credentials=AuthApiKey(config["api_key"]),
            )
            if not self.client.collections.exists(self.collection_name):
                logger.error(f"Collection '{self.collection_name}' does not exist.")
                raise ValueError(f"Collection '{self.collection_name}' not found in Weaviate.")
            logger.info(f"Connected to Weaviate. Using collection '{self.collection_name}'.")
        except ImportError:
            logger.error("Weaviate Python client not installed. Please install it with: pip install weaviate-client")
            raise
        except Exception as e:
            logger.error(f"Error connecting to Weaviate: {e}")
            raise
            
        if azure_config:
            try:
                self.azure_client = AzureOpenAI(
                    azure_endpoint=azure_config["endpoint"],
                    api_key=azure_config["api_key"],
                    api_version=azure_config["api_version"]
                )
                self.embedding_model = azure_config["embedding_deployment"]
                logger.info("Connected to Azure OpenAI service.")
            except Exception as e:
                logger.error(f"Error initializing Azure OpenAI client: {e}")
                raise

    def __del__(self):
        """Ensure Weaviate client is closed when object is destroyed."""
        if self.client:
            try:
                self.client.close()
                logger.info("Weaviate client connection closed.")
            except Exception as e:
                logger.error(f"Error closing Weaviate client: {e}")

    @lru_cache(maxsize=1000)
    async def get_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding vector for input text using Azure OpenAI.
        """
        if not self.azure_client:
            logger.error("Azure OpenAI client not initialized.")
            return None
        try:
            response = await asyncio.to_thread(
                self.azure_client.embeddings.create,
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            return None

    async def search_query_in_collections(self, query_text: str, top_k: int = 5) -> str:
        """
        Search for relevant information in the Weaviate InsuranceDocs collection using vector similarity.
        """
        if not query_text.strip():
            logger.warning("Empty query received.")
            return "I didn't receive any query. How can I help you with insurance?"
            
        if not self.client:
            logger.error("Weaviate client not initialized.")
            return "I'm sorry, I cannot access the insurance data. Please try again later."
        
        try:
            query_embedding = await self.get_text_embedding(query_text)
            if not query_embedding:
                logger.error("Failed to generate query embedding.")
                return "I'm sorry, I couldn't process your query. Please try again."
                
            collection = self.client.collections.get(self.collection_name)
            try:
                result = await asyncio.to_thread(
                    collection.query.near_vector,
                    near_vector=query_embedding,
                    limit=top_k,
                    distance=0.8,
                    return_properties=["title", "content"],
                    return_metadata=weaviate.classes.query.MetadataQuery(distance=True)
                )
                results = result.objects
                logger.info(f"Found {len(results)} results for query: {query_text}")
            except Exception as e:
                logger.error(f"Error searching collection {self.collection_name}: {e}")
                return "I'm sorry, I couldn't find any relevant information. Can you rephrase your query?"
            
            self.memory.add_user_message(query_text)
            
            if not results:
                logger.info("No relevant results found in Weaviate.")
                response = "I couldn't find any information matching your query in our insurance database. Can you provide more details or try a different question?"
            else:
                response = await self.format_results_with_llm(results, query_text)
                logger.info(f"Generated response: {response}")
                
            self.memory.add_ai_message(response)
            return response
            
        except Exception as e:
            logger.error(f"Error in search process: {e}")
            return "I'm sorry, an error occurred while processing your query. Please try again."

    async def format_results_with_llm(self, details: List[Any], user_input: str) -> str:
        """
        Format search results into a coherent plain text response using LLM, strictly using Weaviate data.
        """
        if not self.azure_client:
            logger.warning("Azure client not available, returning raw content.")
            return "Based on your query, I found some information: " + " ".join([d.properties.get('content', 'No content available')[:200] for d in details[:2]])
            
        try:
            details_text = ""
            for i, detail in enumerate(details[:5]):
                title = detail.properties.get('title', 'Untitled')
                content = detail.properties.get('content', 'No content available')
                details_text += f"Document {i+1}: {title}. {content}\n"
            
            chat_history = self.memory.messages
            
            prompt_template = PromptTemplate(
                input_variables=["chat_history", "details", "user_input"],
                template="""
                You are an AI assistant for Major Hong Leong Assurance in Singapore, providing accurate information about insurance products and policies based solely on the provided data from the InsuranceDocs collection.
                Below is the conversation history:
                {chat_history}
                
                Below is the retrieved data from our insurance database:
                {details}
                
                User Query: {user_input}
                
                Provide a helpful response addressing the user's query using only the information in the retrieved data.
                If the data is insufficient, say so and ask for clarification.
                Follow these strict rules:
                - Write in a conversational, human-readable style.
                - Do NOT use markup language (e.g., **, *, -, #, >).
                - Do NOT use bullet points, numbered lists, or headings.
                - Use [BOLD] markers around key point labels (e.g., [BOLD]Type[BOLD], [BOLD]Key Benefits[BOLD]) to highlight them.
                - Use simple sentences with minimal punctuation (only periods, commas, and [BOLD] markers).
                - Keep the tone friendly and approachable.
                - Give a precise and clear information about it 
                - Combine information from multiple documents if relevant, but keep the response concise.
                - Suggest follow-up questions or recommendation queries similar to the chat query and responses after the response.
                Example response: The SMART Saver Plus is a savings plan. [BOLD]Type[BOLD] is endowment. [BOLD]Premium Payment[BOLD] lasts five years. It offers guaranteed payments later. You might want to ask what is the premium payment for the SMART Saver Plus policy or what are the key benefits of the SMART Saver Plus policy.
                Example Query: A 25 years old male residing in Singapore looking for insurance policy?
                Example response: The response should include the query result and recommendation queries like "What is the premium payment for the SMART Saver Plus policy?" or "What are the key benefits of the SMART Saver Plus policy?" or if any requirements are missing, prompt like "Can you please provide your age?" or "Can you please provide your gender?"
                Assistant:
                """
            )
            
            refined_prompt = prompt_template.format(
                chat_history=str(chat_history),
                details=details_text,
                user_input=user_input
            )
            
            response = await asyncio.to_thread(
                self.azure_client.chat.completions.create,
                model=AZURE_CONFIG["gpt_deployment"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI assistant for Major Hong Leong Assurance in Singapore. "
                            "Provide accurate information about insurance products and policies based only on the provided data from the InsuranceDocs collection. "
                            "Always respond in a conversational, human-readable style without markup language, using [BOLD] markers for key point labels. "
                            "Use simple sentences with minimal punctuation. "
                            "Keep the tone friendly and approachable."
                        )
                    },
                    {"role": "user", "content": refined_prompt}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error formatting results: {e}")
            return "I found some information, but I couldn't process it properly. Can you rephrase your query?"

    async def generate_session_summary(self, user_input: str, response: str) -> str:
        """
        Generate a one-line summary of a conversation turn.
        """
        if not self.azure_client:
            logger.warning("Azure client not available, returning default summary.")
            return f"Query about: {user_input[:30]}..."
            
        try:
            prompt = f"""
            Summarize the following user query and assistant response into a single sentence (maximum 50 characters).
            Use a conversational style and focus on the main topic.
            Do not use markup, lists, or headings.
            
            User Query: {user_input}
            Assistant Response: {response}
            
            Example: Asked about SMART Saver Plus features.
            """
            
            response = await asyncio.to_thread(
                self.azure_client.chat.completions.create,
                model=AZURE_CONFIG["gpt_deployment"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI assistant that summarizes conversations. "
                            "Generate a one-line summary (max 50 characters) of the user query and assistant response. "
                            "Use a conversational style and focus on the main topic. "
                            "Do not use markup, lists, or headings."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )
            summary = response.choices[0].message.content.strip()
            if len(summary) > 50:
                summary = summary[:47] + "..."
            logger.info(f"Generated session summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Error generating session summary: {e}")
            return f"Query about: {user_input[:30]}..."

    async def detect_language(self, text: str) -> str:
        """
        Detect the language of the input text.
        """
        if not text.strip():
            return "english"
            
        if not self.azure_client:
            logger.warning("No Azure client available, defaulting to English")
            return "english"
            
        try:
            response = await asyncio.to_thread(
                self.azure_client.chat.completions.create,
                model=AZURE_CONFIG["gpt_deployment"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI assistant that identifies the language of text. "
                            "Return only the language name in lowercase (e.g., english, malay, tamil, chinese, thai, telugu). "
                            "If the text is Singlish or mixed, return 'english'. "
                            "Do not include any other text or explanations"
                        )
                    },
                    {"role": "user", "content": f"Identify the language of the following text: '{text}'"}
                ],
                temperature=0.3
            )
            detected_lang = response.choices[0].message.content.strip().lower()
            
            if detected_lang in LANGUAGES:
                logger.info(f"Detected language: {detected_lang}")
                return detected_lang
                
            logger.warning(f"Detected language '{detected_lang}' not in LANGUAGES, defaulting to english")
            return "english"
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "english"

    async def translate_text(self, text: str, src_lang: str, dest_lang: str) -> str:
        """
        Translate text from source language to destination language.
        """
        if not text.strip():
            return text
            
        if src_lang == dest_lang or not self.azure_client:
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
                        "content": (
                            "You are an AI assistant that translates text accurately. "
                            "Translate the provided text from the source language to the destination language. "
                            "Return only the translated text without any markup, formatting, or additional explanations. "
                            "Use a conversational, human-readable style with minimal punctuation, only using periods and commas where needed. "
                            "Preserve [BOLD] markers in the translated text."
                        )
                    },
                    {"role": "user", "content": f"Translate the following {src_lang_name} text to {dest_lang_name}: '{text}'"}
                ],
                temperature=0.5
            )
            translated_text = response.choices[0].message.content.strip()
            logger.info(f"Translated from {src_lang_name} to {dest_lang_name}")
            return translated_text
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text

    def list_all_documents(self, collection_name: str) -> List[Dict[str, str]]:
        """
        List all documents in a collection.
        """
        if not self.client or not self.client.collections.exists(collection_name):
            logger.error(f"Collection {collection_name} does not exist.")
            return []
            
        try:
            collection = self.client.collections.get(collection_name)
            results = collection.query.fetch_objects(
                return_properties=["title", "content"]
            )
            
            documents = []
            for result in results.objects:
                documents.append({
                    "title": result.properties.get('title', 'Untitled'),
                    "id": result.uuid
                })
                
            return documents
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []

class MultilingualVoiceAssistant:
    def __init__(self):
        """
        Initialize the multilingual voice assistant with speech recognition and knowledge base capabilities.
        """
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 500
        
        self.weaviate_manager = WeaviateManager(
            WEAVIATE_CONFIG,
            AZURE_CONFIG
        )
        
        self.user_language = "en"
        self.display_language = "english"
        self.input_mode = "text"
        self.output_mode = "text"
        
        self.conversation_history = {}
        
        logger.info("Multilingual Voice Assistant initialized")

    def __del__(self):
        """Ensure cleanup of resources when object is destroyed."""
        pass

    def extract_entities(self, message: str) -> Dict[str, Any]:
        """
        Extract entities (age, gender, location) from the user message using regex.
        """
        entities = {"age": None, "gender": None, "location": None}
        
        # Extract age (e.g., "25yr", "25 years old", "25")
        age_match = re.search(r'(\d{1,3})\s*(?:year|yr|years)?\s*(?:old)?', message, re.IGNORECASE)
        if age_match:
            entities["age"] = int(age_match.group(1))
        
        # Extract gender (e.g., "male", "female", "man", "woman")
        gender_match = re.search(r'\b(male|female|man|woman|gentleman|lady)\b', message, re.IGNORECASE)
        if gender_match:
            entities["gender"] = gender_match.group(1).lower()
            if entities["gender"] in ["man", "gentleman"]:
                entities["gender"] = "male"
            elif entities["gender"] in ["woman", "lady"]:
                entities["gender"] = "female"
        
        # Extract location (e.g., "Singapore", "in Singapore")
        location_match = re.search(r'\b(?:in|residing in|from|live in)\s*([A-Za-z\s]+)\b', message, re.IGNORECASE)
        if location_match:
            entities["location"] = location_match.group(1).strip()
        
        logger.debug(f"Extracted entities: {entities}")
        return entities

    def is_recommend_insurance_intent(self, message: str) -> bool:
        """
        Check if the message indicates an intent to get insurance recommendations.
        """
        patterns = [
            r'insurance.*should\s+I',
            r'recommend.*insurance',
            r'what.*insurance.*for\s+me',
            r'best.*insurance'
        ]
        return any(re.search(pattern, message.lower()) for pattern in patterns)

    async def process_user_input(self, text_input=None, session_id=None) -> Tuple[str, Optional[bytes]]:
        """
        Process user input (voice or text) and generate a response, with entity extraction for insurance queries.
        """
        if not session_id:
            session_id = str(int(time.time() * 1000))
        
        # Initialize session if not exists
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = {
                "messages": [],
                "entities": {},
                "pending_intent": None,
                "summary": "Untitled Session"
            }
        
        logger.info(f"Processing user input: {text_input if text_input else 'via voice'} for session {session_id}")
        user_input = None
        audio_bytes = None
        response = None
        summary = None
        
        if self.input_mode == "voice" and text_input is None:
            user_input = self.speech_to_text()
            if not user_input:
                response = "I couldn't hear you. Please try again."
                summary = "Failed to capture voice input."
        else:
            user_input = text_input if text_input else ""
        
        if not user_input or not user_input.strip():
            response = "No input provided. Please try again."
            summary = "Received empty input."
        
        original_input = user_input
        logger.info(f"User input: {user_input}")
        
        detected_lang = await self.weaviate_manager.detect_language(user_input)
        for lang_name, lang_data in LANGUAGES.items():
            if lang_name == detected_lang:
                self.user_language = lang_data["code"]
                self.display_language = lang_name
                break
                
        logger.info(f"Detected language: {detected_lang} (code: {self.user_language})")
        
        if response:  # Early exit for error cases
            if self.user_language != "en":
                response = await self.weaviate_manager.translate_text(response, "en", self.user_language)
            self.conversation_history[session_id]["messages"].append({
                "role": "user",
                "content": original_input or "No input",
                "language": detected_lang,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history[session_id]["messages"].append({
                "role": "assistant",
                "content": response,
                "language": self.display_language,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history[session_id]["summary"] = summary
            return response, None
        
        # Extract entities
        entities = self.extract_entities(user_input)
        for key, value in entities.items():
            if value:
                self.conversation_history[session_id]["entities"][key] = value
        
        # Check for recommend_insurance intent
        user_input_english = user_input
        if detected_lang != "english":
            user_input_english = await self.weaviate_manager.translate_text(
                user_input, 
                LANGUAGES[detected_lang]["code"], 
                "en"
            )
            logger.info(f"Translated to English: {user_input_english}")
        
        if not user_input_english or not user_input_english.strip():
            response = "Failed to process your input. Please try again."
            summary = "Failed to process input."
            if self.user_language != "en":
                response = await self.weaviate_manager.translate_text(response, "en", self.user_language)
            self.conversation_history[session_id]["messages"].append({
                "role": "user",
                "content": original_input,
                "language": detected_lang,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history[session_id]["messages"].append({
                "role": "assistant",
                "content": response,
                "language": self.display_language,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history[session_id]["summary"] = summary
            return response, None
        
        try:
            # Handle recommend_insurance intent
            if self.is_recommend_insurance_intent(user_input_english):
                self.conversation_history[session_id]["pending_intent"] = "recommend_insurance"
                
                if not self.conversation_history[session_id]["entities"].get("age"):
                    response_english = "Please provide your age."
                    summary = "Prompted for age."
                elif not self.conversation_history[session_id]["entities"].get("gender"):
                    response_english = "Please specify your gender."
                    summary = "Prompted for gender."
                else:
                    self.conversation_history[session_id]["pending_intent"] = None
                    response_english = await self.generate_response(user_input_english, session_id)
                    summary = await self.weaviate_manager.generate_session_summary(original_input, response_english)
            elif self.conversation_history[session_id]["pending_intent"] == "recommend_insurance":
                if not self.conversation_history[session_id]["entities"].get("age"):
                    if entities["age"]:
                        self.conversation_history[session_id]["entities"]["age"] = entities["age"]
                        response_english = "Please specify your gender."
                        summary = "Prompted for gender."
                    else:
                        response_english = "Please provide your age."
                        summary = "Prompted for age."
                elif not self.conversation_history[session_id]["entities"].get("gender"):
                    if entities["gender"]:
                        self.conversation_history[session_id]["entities"]["gender"] = entities["gender"]
                        self.conversation_history[session_id]["pending_intent"] = None
                        response_english = await self.generate_response(user_input_english, session_id)
                        summary = await self.weaviate_manager.generate_session_summary(original_input, response_english)
                    else:
                        response_english = "Please specify your gender."
                        summary = "Prompted for gender."
            else:
                response_english = await self.weaviate_manager.search_query_in_collections(user_input_english)
                summary = await self.weaviate_manager.generate_session_summary(original_input, response_english)
            
            response = response_english
            if self.user_language != "en":
                response = await self.weaviate_manager.translate_text(response_english, "en", self.user_language)
                logger.info(f"Translated to {self.display_language}: {response}")
            
            self.conversation_history[session_id]["messages"].append({
                "role": "user",
                "content": original_input,
                "language": detected_lang,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history[session_id]["messages"].append({
                "role": "assistant",
                "content": response,
                "language": self.display_language,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history[session_id]["summary"] = summary
            
            if self.output_mode in ["voice", "both"]:
                output_lang = self.user_language
                supported_tts_langs = ["en", "ms", "th", "zh-CN", "ta", "te"]
                if output_lang not in supported_tts_langs:
                    output_lang = "en"
                    
                logger.info(f"Generating TTS in language: {output_lang}")
                audio_bytes = self.text_to_speech(response, output_lang)
            
            return response, audio_bytes
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            response = "I'm sorry, I encountered an error while processing your request. Please try again."
            summary = "Error processing query."
            if self.user_language != "en":
                response = await self.weaviate_manager.translate_text(response, "en", self.user_language)
            self.conversation_history[session_id]["messages"].append({
                "role": "user",
                "content": original_input,
                "language": detected_lang,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history[session_id]["messages"].append({
                "role": "assistant",
                "content": response,
                "language": self.display_language,
                "timestamp": datetime.now().isoformat()
            })
            self.conversation_history[session_id]["summary"] = summary
            return response, None

    async def generate_response(self, message: str, session_id: str) -> str:
        """
        Generate a response with entity context for insurance recommendations.
        """
        try:
            entities = self.conversation_history[session_id]["entities"]
            context = ""
            if entities.get("age"):
                context += f"User is {entities['age']} years old. "
            if entities.get("gender"):
                context += f"User is {entities['gender']}. "
            if entities.get("location"):
                context += f"User resides in {entities['location']}. "
            
            query = context + message
            response = await self.weaviate_manager.search_query_in_collections(query)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "An error occurred while processing your request. Please try again."

    def text_to_speech(self, text: str, lang_code: str) -> Optional[bytes]:
        """
        Convert text to speech and return audio bytes without saving to a file.
        """
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

    def speech_to_text(self, timeout: int = 15) -> Optional[str]:
        """
        Convert speech to text using the recognizer.
        """
        with sr.Microphone() as source:
            logger.info("Listening for speech input...")
            
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=timeout)
                logger.info("Processing speech...")
                
                google_lang_code = LANGUAGES.get(
                    self.display_language, 
                    {"google_code": "en-US"}
                )["google_code"]
                
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
        """
        Set the input mode (text or voice).
        """
        if mode.lower() in ["text", "voice"]:
            self.input_mode = mode.lower()
            logger.info(f"Input mode set to {self.input_mode}")
            return True
        else:
            logger.warning(f"Invalid input mode: {mode}. Use 'text' or 'voice'.")
            return False
    
    def set_output_mode(self, mode: str) -> bool:
        """
        Set the output mode (text, voice, or both).
        """
        if mode.lower() in ["text", "voice", "both"]:
            self.output_mode = mode.lower()
            logger.info(f"Output mode set to {self.output_mode}")
            return True
        else:
            logger.warning(f"Invalid output mode: {mode}. Use 'text', 'voice', or 'both'.")
            return False
    
    def set_language(self, language: str) -> bool:
        """
        Set the user's preferred language.
        """
        language = language.lower()
        if language in LANGUAGES:
            self.user_language = LANGUAGES[language]["code"]
            self.display_language = language
            logger.info(f"Language set to {language} (code: {self.user_language})")
            return True
        else:
            logger.warning(f"Unsupported language: {language}. Using English.")
            self.user_language = "en"
            self.display_language = "english"
            return False
            
    def get_conversation_history(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the conversation history, organized by session with summaries and entities.
        """
        return self.conversation_history
        
    def clear_conversation_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = {}
        self.weaviate_manager.memory = ChatMessageHistory()
        logger.info("Conversation history cleared")
        
    def save_conversation(self, filename: str) -> bool:
        """
        Save the conversation history to a file.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            logger.info(f"Conversation saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False
            
    async def get_supported_languages(self) -> str:
        """
        Get a list of supported languages in the user's preferred language.
        """
        languages_list = ", ".join([LANGUAGES[lang]["display"] for lang in LANGUAGES])
        response = f"I support the following languages: {languages_list}"
        
        if self.user_language != "en":
            response = await self.weaviate_manager.translate_text(response, "en", self.user_language)
        
        return response
