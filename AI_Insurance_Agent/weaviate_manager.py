import os
import time
import threading
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile
from langdetect import detect
import googletrans
import weaviate
from weaviate.auth import AuthApiKey
from openai import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
import uuid

# Configuration
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
    "api_key": "7WYgbnwGu1U7OoVI2ZUdgAzp9zQ2uLCcpFR5xNCSBIYP9DcwLYRGJQQJ99BEACYeBjFXJ3w3AAABACOGKn3Y",
    "endpoint": "https://idkrag.openai.azure.com/",
    "api_version": "2025-01-01-preview",
    "embedding_deployment": "text-embedding-ada-002",
    "gpt_deployment":  "gpt-4o"
}

LANGUAGES = {
    "english": "en",
    "malay": "ms",
    "thai": "th",
    "chinese": "zh-CN",
    "tamil": "ta",
    "telugu": "te",
    "singlish": "en"
}

class WeaviateManager:
    # def __init__(self, http_host, http_port, grpc_host, grpc_port, secure=False, azure_config=None):
    def __init__(self, weaviate_config=None, azure_config=None):
        self.client =  weaviate.connect_to_wcs(
                cluster_url=WEAVIATE_CONFIG["url"],
                auth_credentials=AuthApiKey(WEAVIATE_CONFIG["api_key"]),
            )
        self.azure_client = None
        self.collection_names = []
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        
        if azure_config:
            self.azure_client = AzureOpenAI(
                azure_endpoint=azure_config["endpoint"],
                api_key=azure_config["api_key"],
                api_version=azure_config["api_version"]  # Fixed parameter name
            )
            self.embedding_model = azure_config["embedding_deployment"]
    
    def get_collection_names(self):
        collections = self.client.collections.list_all()
        self.collection_names = collections
        return self.collection_names

    def search_query_in_collections(self, query_text, top_k=5):
        query_embedding = self.get_text_embedding(query_text) if self.azure_client else None
        results = []
        if not self.collection_names:
            self.get_collection_names()
        for collection_name in self.collection_names:
            if self.client.collections.exists(collection_name):
                collection = self.client.collections.get(collection_name)
                result = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=top_k,
                    distance=0.7,
                    return_properties=["title", "content"]
                )
                results.extend(result.objects)
        self.memory.save_context({"input": query_text}, {"output": ""})
        return self.format_results_with_llm(results, query_text)

    def format_results_with_llm(self, details, user_input):
        chat_history = self.memory.load_memory_variables({})["chat_history"]
        prompt_template = PromptTemplate(
            input_variables=["chat_history", "details", "user_input"],
            template="""
            You are an AI assistant designed to help users understand insurance options based on their needs.
            Below is the conversation history:
            {chat_history}
            Below is the retrieved data:
            {details}
            User Query: {user_input}
            Based on the retrieved information, provide a helpful response addressing the user's query.
            If no relevant information is found, provide general guidance and ask clarifying questions.
            Make your response conversational and easy to understand.
            Assistant:
            """
        )
        refined_prompt = prompt_template.format(
            chat_history=chat_history,
            details=details,
            user_input=user_input
        )
        response = self.azure_client.chat.completions.create(
            model=AZURE_CONFIG["gpt_deployment"],  # Use the deployment name from config
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": refined_prompt}
            ],
            temperature=0.7
        )
        self.memory.save_context({"input": user_input}, {"output": response.choices[0].message.content})
        return response.choices[0].message.content

    def get_text_embedding(self, text):
        if not self.azure_client:
            raise ValueError("Azure OpenAI client not configured.")
        response = self.azure_client.embeddings.create(input=text, model=self.embedding_model)
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

class MultilingualVoiceAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.translator = googletrans.Translator()
        # self.weaviate_manager = WeaviateManager(
        #     WEAVIATE_CONFIG["http_host"],
        #     WEAVIATE_CONFIG["http_port"],
        #     WEAVIATE_CONFIG["grpc_host"],
        #     WEAVIATE_CONFIG["grpc_port"],
        #     WEAVIATE_CONFIG["secure"],
        #     AZURE_CONFIG
        # )
        self.weaviate_manager = WeaviateManager(WEAVIATE_CONFIG, AZURE_CONFIG)
        self.user_language = "en"
        self.input_mode = "text"
        self.output_mode = "text"
        self.temp_dir = tempfile.mkdtemp()
        
    def detect_language(self, text):
        try:
            return detect(text)
        except:
            return "en"
    
    def translate_text(self, text, src_lang, dest_lang):
        if src_lang == dest_lang:
            return text
        try:
            if src_lang not in googletrans.LANGUAGES and src_lang != 'auto':
                src_lang = 'auto'
            if dest_lang not in googletrans.LANGUAGES:
                print(f"Warning: Unsupported destination language '{dest_lang}'. Defaulting to English.")
                dest_lang = 'en'
            translation = self.translator.translate(text, src=src_lang, dest=dest_lang)
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
            return text
    
    def text_to_speech(self, text, lang_code, save_to_file=False):
        try:
            temp_file = os.path.join(self.temp_dir, f"tts_{uuid.uuid4()}.mp3")
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(temp_file)
            if not save_to_file:
                audio = AudioSegment.from_mp3(temp_file)
                play(audio)
                os.remove(temp_file)
            return temp_file
        except Exception as e:
            print(f"TTS error: {e}")
            return None
    
    # New method to capture speech input
    def speech_to_text(self, timeout=5):
        with sr.Microphone() as source:
            print("Listening...")
            try:
                # Adjust for ambient noise and set timeout
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=timeout)
                print("Processing speech...")
                
                # Try to recognize using Google's speech recognition
                text = self.recognizer.recognize_google(audio)
                print(f"Recognized: {text}")
                return text
            except sr.WaitTimeoutError:
                print("No speech detected within timeout period")
                return None
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return None
            except Exception as e:
                print(f"Error in speech recognition: {e}")
                return None
    
    # Method to process user input (either text or speech)
    def process_user_input(self):
        user_input = None
        
        if self.input_mode == "voice":
            user_input = self.speech_to_text()
            if not user_input:
                return "I couldn't hear you. Please try again."
        else:
            user_input = input("You: ")
        
        if not user_input:
            return "No input provided."
        
        # Detect language of input
        detected_lang = self.detect_language(user_input)
        print(f"Detected language: {detected_lang}")
        
        # Translate to English for processing if needed
        if detected_lang != "en":
            english_text = self.translate_text(user_input, detected_lang, "en")
            print(f"Translated to English: {english_text}")
            response_english = self.weaviate_manager.search_query_in_collections(english_text)
            
            # Translate response back to user's language
            response = self.translate_text(response_english, "en", detected_lang)
        else:
            response = self.weaviate_manager.search_query_in_collections(user_input)
        
        # Output response
        if self.output_mode == "voice":
            output_lang = detected_lang if detected_lang in googletrans.LANGUAGES else "en"
            self.text_to_speech(response, output_lang)
        
        return response
    
    # Method to set input mode
    def set_input_mode(self, mode):
        if mode.lower() in ["text", "voice"]:
            self.input_mode = mode.lower()
            print(f"Input mode set to {self.input_mode}")
        else:
            print("Invalid input mode. Use 'text' or 'voice'.")
    
    # Method to set output mode
    def set_output_mode(self, mode):
        if mode.lower() in ["text", "voice"]:
            self.output_mode = mode.lower()
            print(f"Output mode set to {self.output_mode}")
        else:
            print("Invalid output mode. Use 'text' or 'voice'.")
    
    # Method to set user language
    def set_language(self, language):
        if language.lower() in LANGUAGES:
            self.user_language = LANGUAGES[language.lower()]
            print(f"Language set to {language}")
        else:
            print(f"Unsupported language: {language}. Using English.")
            self.user_language = "en"
    
    # Main interaction loop
    def run(self):
        print("Welcome to the Multilingual Voice Assistant!")
        print("Available commands:")
        print("  'input text/voice' - Change input mode")
        print("  'output text/voice' - Change output mode")
        print("  'language [language]' - Set language (english, malay, thai, chinese, tamil, telugu)")
        print("  'exit' - Exit the assistant")
        
        running = True
        while running:
            try:
                if self.input_mode == "text":
                    command = input("\nEnter command or question: ").strip().lower()
                else:
                    print("\nSpeak a command or question...")
                    command = self.speech_to_text()
                    if not command:
                        print("I didn't catch that. Please try again.")
                        continue
                    print(f"You said: {command}")
                    command = command.strip().lower()
                
                if command == "exit":
                    running = False
                    print("Goodbye!")
                elif command.startswith("input "):
                    mode = command.split(" ")[1]
                    self.set_input_mode(mode)
                elif command.startswith("output "):
                    mode = command.split(" ")[1]
                    self.set_output_mode(mode)
                elif command.startswith("language "):
                    lang = command.split(" ")[1]
                    self.set_language(lang)
                else:
                    response = self.process_user_input()
                    print(f"\nAssistant: {response}")
            
            except KeyboardInterrupt:
                running = False
                print("\nExiting...")
            except Exception as e:
                print(f"Error: {e}")

# Run the assistant if this file is executed directly
if __name__ == "__main__":
    assistant = MultilingualVoiceAssistant()
    assistant.run()