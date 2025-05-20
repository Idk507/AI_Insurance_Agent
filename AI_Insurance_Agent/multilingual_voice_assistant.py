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

# Configuration
WEAVIATE_CONFIG = {
    "http_host": "localhost",
    "http_port": 8080,
    "grpc_host": "localhost",
    "grpc_port": 50051,
    "secure": False,
    "class_name": "InsuranceDocs"
}

AZURE_CONFIG = {
    "api_key": "7WYgbnwGu1U7OoVI2ZUdgAzp9zQ2uLCcpFR5xNCSBIYP9DcwLYRGJQQJ99BEACYeBjFXJ3w3AAABACOGKn3Y",
    "endpoint": "https://idkrag.openai.azure.com/",
    "api_version": "2025-01-01-preview",
    "embedding_deployment": "text-embedding-ada-002",
    "gpt_deployment":  "gpt-4o"
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

class WeaviateManager:
    def __init__(self, config: Dict, azure_config: Dict):
        self.client = None
        self.collection_name = config["class_name"]
        self.azure_client = None
        self.memory = ChatMessageHistory()
        self.embedding_model = azure_config["embedding_deployment"]

        try:
            self.client = weaviate.connect_to_custom(
                http_host=config["http_host"],
                http_port=config["http_port"],
                grpc_host=config["grpc_host"],
                grpc_port=config["grpc_port"],
                http_secure=config["secure"],
                grpc_secure=config["secure"]
            )
            if not self.client.collections.exists(self.collection_name):
                logger.error(f"Collection '{self.collection_name}' does not exist.")
                raise ValueError(f"Collection '{self.collection_name}' not found in Weaviate.")
            logger.info(f"Connected to Weaviate. Using collection '{self.collection_name}'.")
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
            result = await asyncio.to_thread(
                collection.query.near_vector,
                near_vector=query_embedding,
                limit=top_k,
                distance=0.8,
                return_properties=["title", "content"],
                return_metadata=wvc.query.MetadataQuery(distance=True)
            )
            results = result.objects
            logger.info(f"Found {len(results)} results for query: {query_text}")
            
            self.memory.add_user_message(query_text)
            
            if not results:
                logger.info("No relevant results found in Weaviate.")
                return "I couldn't find any information matching your query in our insurance database. Can you provide more details or try a different question?"
            else:
                response = await self.format_results_with_llm(results, query_text)
                logger.info(f"Generated response: {response}")
                
            self.memory.add_ai_message(response)
            return response
            
        except Exception as e:
            logger.error(f"Error in search process: {e}")
            return "I'm sorry, an error occurred while processing your query. Please try again."

    async def format_results_with_llm(self, details: List, user_input: str) -> str:
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
                - Use [BOLD] markers around key point labels (e.g., [BOLD]Type[BOLD], [BOLD]Key Benefits[BOLD]).
                - Use simple sentences with minimal punctuation (only periods, commas, and [BOLD]markers).
                - Keep the tone friendly and approachable.
                - Combine information from multiple documents if relevant, but keep the response concise.
                - Suggest follow-up questions or recommendation queries similar to the chat query and responses after the response.
                Example response: The SMART Saver Plus is a savings plan. [BOLD]Type[BOLD] is endowment. [BOLD]Premium Payment [BOLD]lasts five years. It offers guaranteed payments later. You might want to ask what is the premium payment for the SMART Saver Plus policy or what are the key benefits of the SMART Saver Plus policy.
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

    async def detect_language(self, text: str) -> str:
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
            with open('conversation_history.json', 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            logger.info("Loaded conversation history from file")
        except FileNotFoundError:
            logger.info("No saved conversation history found")
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")

    def save_conversation(self, filename='conversation_history.json'):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            logger.info(f"Conversation saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return False

    async def process_user_input(self, text_input: str, session_id: str, language: str = "english", input_mode: str = "text", output_mode: str = "text"):
        try:
            # Update language and modes
            self.set_language(language)
            self.set_input_mode(input_mode)
            self.set_output_mode(output_mode)

            # Detect language if input is voice
            if self.input_mode == "voice":
                detected_lang = await self.weaviate_manager.detect_language(text_input)
                self.set_language(detected_lang)
                logger.info(f"Detected language for voice input: {detected_lang}")

            # Extract entities and check intent
            entities = self.extract_entities(text_input)
            is_recommendation = self.is_recommend_insurance_intent(text_input)

            # Generate response using Weaviate search
            response = await self.generate_response(text_input, session_id, entities, is_recommendation)
            
            # Translate response if needed
            if self.user_language != "en":
                response = await self.weaviate_manager.translate_text(response, "en", self.user_language)
            
            # Generate suggestions
            suggestions = await self.generate_suggestions(text_input, response, session_id)
            
            # Generate audio if needed
            audio_data = None
            if self.output_mode in ["voice", "both"]:
                audio_data = await self.generate_audio(response)
            
            # Store in history
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = {"messages": [], "summary": "", "entities": {}, "pending_intent": None}
            timestamp = datetime.now().isoformat()
            self.conversation_history[session_id]["messages"].append({
                "role": "user",
                "content": text_input,
                "timestamp": timestamp,
                "language": self.display_language
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
            self.conversation_history[session_id]["summary"] = await self.generate_session_summary(session_id)
            self.save_conversation()
            
            logger.info(f"Processed input: {text_input}, Response: {response}, Suggestions: {suggestions}")
            return response, audio_data, suggestions
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            raise

    async def generate_response(self, text_input: str, session_id: str, entities: Dict, is_recommendation: bool) -> str:
        try:
            if not self.weaviate_manager.azure_client:
                logger.error("Azure OpenAI client not initialized.")
                return "Sorry, I couldn't process your request due to a configuration issue."
            
            # Enhance query with entities for recommendations
            enhanced_query = text_input
            if is_recommendation and entities:
                entity_info = []
                if entities["age"]:
                    entity_info.append(f"age {entities['age']}")
                if entities["gender"]:
                    entity_info.append(f"{entities['gender']}")
                if entities["location"]:
                    entity_info.append(f"residing in {entities['location']}")
                if entity_info:
                    enhanced_query += f" (for someone {', '.join(entity_info)})"
            
            # Use Weaviate search for insurance-related queries
            response = await self.weaviate_manager.search_query_in_collections(enhanced_query)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I couldn't process your request."

    async def generate_suggestions(self, user_input: str, response: str, session_id: str) -> List[str]:
        try:
            if not self.weaviate_manager.azure_client:
                logger.error("Azure OpenAI client not initialized.")
                return []
                
            # Get recent conversation history for context
            session = self.conversation_history.get(session_id, {})
            messages = session.get("messages", [])[-6:]  # Last 3 exchanges (user + assistant)
            history_text = json.dumps(messages, indent=2)
            
            # Updated prompt to enforce JSON output and use history
            prompt = f"""
            You are an AI assistant for Major Hong Leong Assurance in Singapore.
            Based on the conversation history, user query, and response, generate 2-3 relevant follow-up questions the user might ask next.
            Use the conversation history to avoid repeating previous questions and ensure suggestions are contextually relevant.
            Return the questions as a JSON array of strings, with no additional text, code fences, or explanations.
            Conversation history: {history_text}
            User query: "{user_input}"
            Response: "{response}"
            Example output: ["What does car insurance cover?", "How much is car insurance?"]
            """
            
            completion = await asyncio.to_thread(
                self.weaviate_manager.azure_client.chat.completions.create,
                model=AZURE_CONFIG["gpt_deployment"],
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI assistant that generates follow-up questions. "
                            "Return only a JSON array of 2-3 strings, with no additional text, code fences, or explanations. "
                            "Ensure questions are relevant, unique, and based on the provided conversation history, user query, and response."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.7
            )
            
            raw_response = completion.choices[0].message.content.strip()
            logger.info(f"Raw suggestion response: {raw_response}")
            
            # Attempt to parse JSON
            try:
                suggestions = json.loads(raw_response)
                if not isinstance(suggestions, list) or not all(isinstance(s, str) for s in suggestions):
                    logger.warning("Response is not a list of strings, attempting to extract JSON array")
                    raise ValueError("Invalid JSON format")
                suggestions = suggestions[:3]  # Limit to 3 suggestions
                logger.info(f"Parsed suggestions: {suggestions}")
                return suggestions
            except json.JSONDecodeError:
                # Try to extract JSON array from response (e.g., if wrapped in code fences)
                json_match = re.search(r'\[.*?\]', raw_response, re.DOTALL)
                if json_match:
                    try:
                        suggestions = json.loads(json_match.group(0))
                        if isinstance(suggestions, list) and all(isinstance(s, str) for s in suggestions):
                            suggestions = suggestions[:3]
                            logger.info(f"Extracted suggestions from regex: {suggestions}")
                            return suggestions
                    except json.JSONDecodeError:
                        pass
                # Fallback to default suggestions
                logger.warning("Failed to parse suggestions, returning default suggestions")
                return [
                    "Can you tell me more about this insurance plan?",
                    "What are the premium rates for this policy?",
                    "Are there any optional covers available?"
                ]
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return [
                "Can you tell me more about this insurance plan?",
                "What are the premium rates for this policy?",
                "Are there any optional covers available?"
            ]

    async def generate_session_summary(self, session_id: str) -> str:
        try:
            if not self.weaviate_manager.azure_client:
                logger.error("Azure OpenAI client not initialized.")
                return "Summary unavailable."
                
            messages = self.conversation_history.get(session_id, {}).get("messages", [])
            if not messages:
                return "No conversation yet."
                
            prompt = f"""
            Summarize the following conversation in 1-2 sentences:
            {json.dumps(messages, indent=2)}
            """
            completion = await asyncio.to_thread(
                self.weaviate_manager.azure_client.chat.completions.create,
                model=AZURE_CONFIG["gpt_deployment"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.7
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
            r'best.*insurance'
        ]
        return any(re.search(pattern, message.lower()) for pattern in patterns)

    def clear_conversation_history(self) -> None:
        self.conversation_history = {}
        self.weaviate_manager.memory = ChatMessageHistory()
        logger.info("Conversation history cleared")