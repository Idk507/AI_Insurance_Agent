import os
import re
import json
import logging
import asyncio
import io
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Form, Request, Depends
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AzureOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate
from gtts import gTTS
import speech_recognition as sr
import weaviate
import weaviate.classes as wvc
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdf
import docx
import tempfile
from data_uploader import DataUploader  
from multilingual_voice_assistant import MultilingualVoiceAssistant
from weaviate.auth import AuthApiKey
from msal_auth import build_auth_url, acquire_token_by_auth_code, get_token_from_cache, remove_account, TENANT_ID, initiate_auth_flow, acquire_token_by_auth_code_flow
import secrets
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging with consistent format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("insurance_assistant.log")
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add session middleware for storing user sessions
# Generate a secure session secret or use from environment
def get_or_create_session_secret():
    """Get session secret from environment or create/load from file."""
    session_secret_file = "session_secret.txt"
    
    # First try environment variable
    env_secret = os.getenv("SESSION_SECRET")
    if env_secret:
        logger.info("Using session secret from environment variable")
        return env_secret
    
    # Try to load from file
    if os.path.exists(session_secret_file):
        try:
            with open(session_secret_file, 'r') as f:
                secret = f.read().strip()
                if secret:
                    logger.info("Loaded session secret from file")
                    return secret
        except Exception as e:
            logger.warning(f"Failed to read session secret from file: {e}")
    
    # Generate new secret and save it
    secret = secrets.token_urlsafe(32)
    try:
        with open(session_secret_file, 'w') as f:
            f.write(secret)
        logger.info("Generated new session secret and saved to file")
    except Exception as e:
        logger.warning(f"Failed to save session secret to file: {e}")
    
    return secret

SESSION_SECRET = get_or_create_session_secret()

app.add_middleware(
    SessionMiddleware, 
    secret_key=SESSION_SECRET,
    max_age=1209600,  # 14 days
    same_site="lax",
    https_only=False  # Set to True in production with HTTPS
)

# Initialize assistant and DataUploader
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
    "collection_names": ["CustomerProfiles", "InsuranceDocs"]
}
try:
    logger.info("Starting initialization of assistant and DataUploader")
    assistant = MultilingualVoiceAssistant()
    data_uploader = DataUploader(WEAVIATE_CONFIG, AZURE_CONFIG)
    for collection_name in WEAVIATE_CONFIG["collection_names"]:
        logger.info(f"Creating collection: {collection_name}")
        data_uploader.create_collection(collection_name, f"Collection for storing {'customer profile' if collection_name == 'CustomerProfiles' else 'insurance document'} data")
    logger.info("Successfully initialized assistant and DataUploader")
except Exception as e:
    logger.error(f"Failed to initialize assistant and DataUploader: {str(e)}")
    raise

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    language: str
    input_mode: str
    output_mode: str
    session_id: str
    customer_id: Optional[str] = None

class ModeRequest(BaseModel):
    mode: str
    session_id: str

class LanguageRequest(BaseModel):
    language: str
    session_id: str

def escape_html(text):
    logger.debug(f"Escaping HTML for text: {text[:100]}...")
    if not text:
        logger.debug("Text is empty, returning as-is")
        return text
    replacements = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    logger.debug(f"Escaped HTML text: {text[:100]}...")
    return text

def format_bold_tags(content):
    logger.debug(f"Formatting bold tags for content: {content[:100]}...")
    if not content:
        logger.debug("Content is empty, returning as-is")
        return content
    parts = []
    current = ""
    is_bold = False
    i = 0
    while i < len(content):
        if content[i:i+6].upper() == "[BOLD]":
            if current:
                parts.append((is_bold, current))
                current = ""
            is_bold = True
            i += 6
        elif content[i:i+7].upper() == "[/BOLD]":
            if current:
                parts.append((is_bold, current))
                current = ""
            is_bold = False
            i += 7
        else:
            current += content[i]
            i += 1
    if current:
        parts.append((is_bold, current))
    logger.debug(f"Formatted bold tags, resulting parts count: {len(parts)}")
    return parts

# Add authentication check dependency functions here, before the routes
def get_current_user_optional(request: Request) -> Optional[Dict]:
    """Get current user from session, returns None if not authenticated."""
    user = request.session.get("user")
    logger.debug(f"get_current_user_optional: user={user is not None}")
    if user:
        logger.debug(f"User details: name={user.get('name')}, email={user.get('email')}")
    return user

def require_auth(request: Request) -> Dict:
    """Require authentication, raises HTTPException if not authenticated."""
    user = request.session.get("user")
    logger.debug(f"require_auth: user={user is not None}")
    if not user:
        logger.warning("Authentication required but user not found in session")
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

# Update the home route to handle authentication
@app.get("/", name="home")
async def serve_home(request: Request, user: Optional[Dict] = Depends(get_current_user_optional)):
    logger.info("Received request to serve home page")
    logger.debug(f"Session data: {dict(request.session)}")
    logger.debug(f"User authenticated: {user is not None}")
    
    try:
        # Check if user is authenticated
        if not user:
            # Serve login page if not authenticated
            logger.info("User not authenticated, serving login page")
            return FileResponse("templates/login.html")
        
        logger.info(f"User authenticated: {user.get('name', 'Unknown')} ({user.get('email', 'No email')})")
        file_path = "templates/home.html"
        if not os.path.exists(file_path):
            logger.error(f"Home page file not found at {file_path}")
            raise HTTPException(status_code=404, detail="Home page not found")
        logger.info("Successfully serving home.html to authenticated user")
        return FileResponse(file_path)
    except Exception as e:
        logger.error(f"Error serving home page: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving home page: {str(e)}")

@app.get("/{page}.html")
async def serve_page(page: str, request: Request, user: Optional[Dict] = Depends(get_current_user_optional)):
    logger.info(f"Received request to serve page: {page}")
    try:
        valid_pages = ["home", "settings", "conversations", "knowledge", "insights", "product_qa", "customer_details", "login"]
        if page not in valid_pages:
            logger.warning(f"Invalid page requested: {page}")
            raise HTTPException(status_code=404, detail="Page not found")
        
        # Check authentication for protected pages
        if page != "login" and not user:
            logger.info(f"User not authenticated, redirecting to login")
            return RedirectResponse(url="/", status_code=302)
        
        file_path = f"templates/{page}.html"
        if not os.path.exists(file_path):
            logger.error(f"Page file not found at {file_path}")
            raise HTTPException(status_code=404, detail=f"File {page}.html not found")
        logger.info(f"Successfully serving {page}.html")
        return FileResponse(file_path)
    except Exception as e:
        logger.error(f"Error serving page {page}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    logger.info(f"Received chat request for session {request.session_id}")
    logger.debug(f"Chat request details: {request.dict()}")
    try:
        if not request.session_id:
            logger.warning("Missing session_id in chat request")
            raise HTTPException(status_code=400, detail="Session ID is required")
        if not request.message.strip():
            logger.warning("Empty message in chat request")
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        logger.info(f"Processing message with context for session {request.session_id}")
        is_product_qa = request.session_id.startswith("product_qa_")
        context = "product_qa" if is_product_qa else "customer" if request.customer_id else "general"
        modified_message = f"[Context: {context}] {request.message}"
        logger.debug(f"Modified message with context: {modified_message}")

        logger.info(f"Calling process_user_input for session {request.session_id}")
        response, audio_data, suggestions = await assistant.process_user_input(
            text_input=modified_message,
            session_id=request.session_id,
            language=request.language,
            input_mode=request.input_mode,
            output_mode=request.output_mode,
            customer_id=request.customer_id if request.customer_id else None
        )
        logger.info(f"Generated response for session {request.session_id} (context: {context}): {response[:100]}...")
        logger.debug(f"Generated suggestions: {suggestions}")
        logger.debug(f"Audio data generated: {bool(audio_data)}")

        response_dict = {
            "response": response,
            "audio_data": audio_data.hex() if isinstance(audio_data, bytes) else None,
            "suggestions": suggestions or []
        }
        logger.info(f"Returning chat response for session {request.session_id}")
        return response_dict
    except HTTPException as e:
        logger.error(f"HTTPException in chat endpoint for session {request.session_id}: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error processing chat request for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/api/chat_history")
async def get_chat_history(session_id: str = None):
    logger.info(f"Received request to get chat history for session: {session_id if session_id else 'all sessions'}")
    try:
        history = assistant.get_conversation_history()
        logger.debug(f"Retrieved chat history with {len(history)} sessions")
        if session_id:
            if session_id not in history:
                logger.warning(f"Session {session_id} not found in chat history")
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            history = {session_id: history.get(session_id, {"messages": [], "summary": "", "entities": {}})}
        logger.info(f"Returning chat history with {len(history)} sessions")
        return {"sessions": history}
    except HTTPException as e:
        logger.error(f"HTTPException in get_chat_history endpoint: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error retrieving chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@app.post("/api/set_input_mode")
async def set_input_mode(request: ModeRequest):
    logger.info(f"Received request to set input mode for session {request.session_id}")
    logger.debug(f"Input mode request: {request.dict()}")
    try:
        if not request.session_id:
            logger.warning("Missing session_id in set_input_mode request")
            raise HTTPException(status_code=400, detail="Session ID is required")
        success = assistant.set_input_mode(request.mode)
        if not success:
            logger.warning(f"Invalid input mode: {request.mode}")
            raise HTTPException(status_code=400, detail="Invalid input mode")
        logger.info(f"Successfully set input mode to {request.mode} for session {request.session_id}")
        return {"status": "success"}
    except HTTPException as e:
        logger.error(f"HTTPException in set_input_mode endpoint: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error setting input mode for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting input mode: {str(e)}")

@app.post("/api/set_output_mode")
async def set_output_mode(request: ModeRequest):
    logger.info(f"Received request to set output mode for session {request.session_id}")
    logger.debug(f"Output mode request: {request.dict()}")
    try:
        if not request.session_id:
            logger.warning("Missing session_id in set_output_mode request")
            raise HTTPException(status_code=400, detail="Session ID is required")
        success = assistant.set_output_mode(request.mode)
        if not success:
            logger.warning(f"Invalid output mode: {request.mode}")
            raise HTTPException(status_code=400, detail="Invalid output mode")
        logger.info(f"Successfully set output mode to {request.mode} for session {request.session_id}")
        return {"status": "success"}
    except HTTPException as e:
        logger.error(f"HTTPException in set_output_mode endpoint: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error setting output mode for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting output mode: {str(e)}")

@app.post("/api/set_language")
async def set_language(request: LanguageRequest):
    logger.info(f"Received request to set language for session {request.session_id}")
    logger.debug(f"Language request: {request.dict()}")
    try:
        if not request.session_id:
            logger.warning("Missing session_id in set_language request")
            raise HTTPException(status_code=400, detail="Session ID is required")
        success = assistant.set_language(request.language)
        if not success:
            logger.warning(f"Invalid language: {request.language}")
            raise HTTPException(status_code=400, detail="Invalid language")
        logger.info(f"Successfully set language to {request.language} for session {request.session_id}")
        return {"status": "success"}
    except HTTPException as e:
        logger.error(f"HTTPException in set_language endpoint: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error setting language for session {request.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting language: {str(e)}")

@app.get("/api/download_chat")
async def download_chat(session_id: str):
    logger.info(f"Received request to download chat for session {session_id}")
    try:
        if not session_id:
            logger.warning("Missing session_id in download_chat request")
            raise HTTPException(status_code=400, detail="Session ID is required")

        history = assistant.get_conversation_history()
        session = history.get(session_id, {"messages": []})
        if not session:
            logger.warning(f"Session {session_id} not found for PDF download")
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        messages = session.get("messages", [])
        logger.debug(f"Found {len(messages)} messages for session {session_id}")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=72, bottomMargin=72, leftMargin=72, rightMargin=72)
        elements = []

        title_style = ParagraphStyle(name='Title', fontSize=16, leading=20, textColor=colors.navy, spaceAfter=12)
        user_style = ParagraphStyle(name='User', fontSize=12, leading=14, textColor=colors.darkblue, spaceBefore=6, spaceAfter=2)
        assistant_style = ParagraphStyle(name='Assistant', fontSize=12, leading=14, textColor=colors.blue, spaceBefore=6, spaceAfter=2)
        assistant_bold_style = ParagraphStyle(name='AssistantBold', fontSize=12, leading=14, textColor=colors.blue, fontName='Helvetica-Bold', spaceBefore=6, spaceAfter=2)
        timestamp_style = ParagraphStyle(name='Timestamp', fontSize=10, leading=12, textColor=colors.grey, spaceAfter=6)
        suggestion_style = ParagraphStyle(name='Suggestion', fontSize=10, leading=12, textColor=colors.darkcyan, leftIndent=20)

        title = "AI Insurance Agent Product Q&A Session" if session_id.startswith("product_qa_") else "AI Insurance Agent Chat Session"
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 12))

        logger.info(f"Processing {len(messages)} messages for PDF generation")
        for msg in messages:
            timestamp = msg.get("timestamp", datetime.now().isoformat())
            content = escape_html(msg.get("content", "") or "No content")
            formatted_parts = format_bold_tags(content)

            if msg["role"] == "user":
                formatted_text = "<b>User:</b> " + "".join([f"<b>{part}</b>" if is_bold else part for is_bold, part in formatted_parts])
                elements.append(Paragraph(formatted_text, user_style))
                elements.append(Paragraph(f"{datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')}", timestamp_style))
            elif msg["role"] == "assistant":
                formatted_text = "<b>Assistant:</b> "
                for is_bold, part in formatted_parts:
                    style = assistant_bold_style if is_bold else assistant_style
                    elements.append(Paragraph(formatted_text + part, style))
                    formatted_text = ""
                elements.append(Paragraph(f"{datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')}", timestamp_style))
                if msg.get("suggestions", []):
                    suggestion_items = [
                        ListItem(Paragraph(escape_html(s), suggestion_style))
                        for s in msg["suggestions"] if s.strip()
                    ]
                    if suggestion_items:
                        elements.append(Paragraph("<b>Suggestions:</b>", assistant_style))
                        elements.append(ListFlowable(suggestion_items, bulletType='bullet', start='circle'))
                        elements.append(Spacer(1, 6))

        if not elements:
            elements.append(Paragraph("No messages in this session.", user_style))
        logger.info("Building PDF document")
        doc.build(elements)
        pdf_data = buffer.getvalue()
        buffer.close()
        logger.info(f"PDF generated successfully for session {session_id}")

        return StreamingResponse(
            io.BytesIO(pdf_data),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=chat_session_{session_id}.pdf"}
        )
    except HTTPException as e:
        logger.error(f"HTTPException in download_chat endpoint for session {session_id}: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error generating PDF for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

@app.post("/api/upload_document")
async def upload_document(file: UploadFile = File(...), category: str = Form(...)):
    logger.info(f"Received request to upload document: {file.filename}, category: {category}")
    try:
        if not file.filename.lower().endswith(('.pdf', '.doc', '.docx', '.csv')):
            logger.warning(f"Invalid file type uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF, Word, and CSV documents are supported")

        if category not in ["customer_profile", "insurance_document"]:
            logger.warning(f"Invalid category provided: {category}")
            raise HTTPException(status_code=400, detail="Invalid category. Must be 'customer_profile' or 'insurance_document'")

        content = ""
        file_bytes = await file.read()
        logger.debug(f"Read {len(file_bytes)} bytes from file {file.filename}")
        collection_name = "CustomerProfiles" if category == "customer_profile" else "InsuranceDocs"
        logger.info(f"Selected collection: {collection_name} for category {category}")

        if file.filename.lower().endswith('.csv'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
                logger.debug(f"Created temporary file at {tmp_file_path}")
            try:
                logger.info(f"Attempting to upload CSV to {collection_name} using DataUploader")
                if category != "customer_profile":
                    logger.warning(f"CSV files are only supported for customer_profile category, not {category}")
                    raise HTTPException(status_code=400, detail="CSV files are only supported for customer_profile category")
                result = data_uploader.add_customer_profile_csv(collection_name, tmp_file_path)
                logger.info(f"Uploaded {file.filename} to {collection_name}: {result}")
                return {
                    "message": f"Document '{file.filename}' uploaded successfully to {collection_name}",
                    **result
                }
            finally:
                logger.debug(f"Cleaning up temporary file at {tmp_file_path}")
                os.unlink(tmp_file_path)

        elif file.filename.lower().endswith('.pdf'):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
                logger.debug(f"Created temporary file at {tmp_file_path}")
            try:
                logger.info(f"Attempting to upload PDF to {collection_name} using DataUploader")
                result = data_uploader.add_insurance_document_pdf(collection_name, tmp_file_path)
                logger.info(f"Uploaded {file.filename} to {collection_name}: {result}")
                return {
                    "message": f"Document '{file.filename}' uploaded successfully to {collection_name}",
                    **result
                }
            except Exception as e:
                logger.warning(f"DataUploader failed for {file.filename}: {str(e)}")
                content = ""
            finally:
                logger.debug(f"Cleaning up temporary file at {tmp_file_path}")
                os.unlink(tmp_file_path)

            if not content.strip():
                logger.info(f"Falling back to pypdf for text extraction from {file.filename}")
                pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text()
                logger.info(f"pypdf extracted {len(content)} characters from {file.filename}")

            if not content.strip():
                logger.warning(f"No text extracted from {file.filename}. Storing metadata only.")
                content = f"Metadata for {file.filename}"
                embedding = data_uploader.get_text_embedding(content)
                if not embedding:
                    logger.error(f"Failed to generate embedding for {file.filename} metadata")
                    raise HTTPException(status_code=500, detail="Failed to generate document embedding")
                collection = data_uploader.client.collections.get(collection_name)
                logger.info(f"Inserting metadata for {file.filename} into {collection_name}")
                collection.data.insert({
                    "title": file.filename,
                    "content": content,
                    "category": category
                }, vector=embedding)
                logger.info(f"Successfully stored metadata for {file.filename}")
                return {
                    "message": f"Document '{file.filename}' uploaded successfully (no text extracted, metadata stored)",
                    "warning": "No text could be extracted from the PDF. It may be scanned or encrypted."
                }

        elif file.filename.lower().endswith(('.doc', '.docx')):
            logger.info(f"Extracting text from Word document {file.filename}")
            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                content += para.text + "\n"
            logger.info(f"Extracted {len(content)} characters from {file.filename}")

        if not content.strip():
            logger.warning(f"Document is empty or unreadable: {file.filename}")
            raise HTTPException(status_code=400, detail="Document is empty or unreadable")

        logger.info(f"Generating embedding for document content")
        embedding = data_uploader.get_text_embedding(content)
        if not embedding:
            logger.error(f"Failed to generate embedding for {file.filename}")
            raise HTTPException(status_code=500, detail="Failed to generate document embedding")
        collection = data_uploader.client.collections.get(collection_name)
        logger.info(f"Inserting document {file.filename} into {collection_name}")
        collection.data.insert({
            "title": file.filename,
            "content": content,
            "category": category
        }, vector=embedding)

        logger.info(f"Successfully uploaded document: {file.filename} to {collection_name}")
        return {"message": f"Document '{file.filename}' uploaded successfully to {collection_name}"}
    except HTTPException as e:
        logger.error(f"HTTPException in upload_document endpoint: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error uploading document {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/api/list_documents")
async def list_documents(collection_name: str = None):
    logger.info(f"Received request to list documents for collection: {collection_name if collection_name else 'all collections'}")
    try:
        collections = [collection_name] if collection_name in WEAVIATE_CONFIG["collection_names"] else WEAVIATE_CONFIG["collection_names"]
        documents = []
        for collection in collections:
            logger.info(f"Listing documents in collection: {collection}")
            docs = data_uploader.list_all_documents(collection)
            logger.debug(f"Retrieved {len(docs)} documents from {collection}")
            for doc in docs:
                doc["collection"] = collection
            documents.extend(docs)
        logger.info(f"Returning {len(documents)} documents from {', '.join(collections)}")
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/api/delete_document/{document_id}")
async def delete_document(document_id: str, collection_name: str = None):
    logger.info(f"Received request to delete document {document_id} from collection: {collection_name if collection_name else 'all collections'}")
    try:
        collections = [collection_name] if collection_name in WEAVIATE_CONFIG["collection_names"] else WEAVIATE_CONFIG["collection_names"]
        for collection in collections:
            try:
                logger.info(f"Attempting to delete document {document_id} from {collection}")
                collection_obj = data_uploader.client.collections.get(collection)
                result = collection_obj.data.delete_by_id(uuid=document_id)
                if result:
                    logger.info(f"Successfully deleted document {document_id} from {collection}")
                    return {"status": "success", "message": f"Document {document_id} deleted successfully from {collection}"}
            except Exception as e:
                logger.warning(f"Document {document_id} not found in {collection}: {str(e)}")
        logger.warning(f"Document {document_id} not found in any collection")
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    except HTTPException as e:
        logger.error(f"HTTPException in delete_document endpoint: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.post("/api/clear_history")
async def clear_history():
    logger.info("Received request to clear conversation history")
    try:
        assistant.clear_conversation_history()
        if os.path.exists("conversation_history.json"):
            logger.info("Deleting conversation history file")
            os.remove("conversation_history.json")
            logger.info("Conversation history file deleted")
        logger.info("Conversation history cleared successfully")
        return {"message": "Conversation history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

@app.get("/api/customers")
async def get_customers(customer_id: Optional[str] = None):
    logger.info(f"Received request to get customers, customer_id: {customer_id if customer_id else 'all customers'}")
    try:
        logger.info("Fetching customer profiles from Weaviate")
        collection = data_uploader.client.collections.get("CustomerProfiles")
        result = collection.query.fetch_objects(
            limit=1000,  # Set a high limit to fetch all customers
            filters=wvc.query.Filter.by_property("category").equal("customer_profile"),
            return_properties=["title", "content"]
        )
        logger.debug(f"Retrieved {len(result.objects)} customer profiles from Weaviate")

        customers = []
        for obj in result.objects:
            try:
                # Log the raw content for debugging
                raw_content = obj.properties["content"]
                logger.debug(f"Processing customer {obj.properties['title']}, raw content: {raw_content[:100]}...")

                # Parse the content as JSON and clean the keys
                raw_details = json.loads(raw_content)
                details = {key.strip(): value for key, value in raw_details.items()}
                logger.debug(f"Parsed and cleaned JSON content for customer {obj.properties['title']}")

                # Construct the customer profile with the cleaned data
                customer = {
                    "customer_id": obj.properties["title"],
                    "details": details  # Include all cleaned fields
                }
                customers.append(customer)
                logger.debug(f"Added customer {obj.properties['title']} to list")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse details for customer {obj.properties['title']}: {str(e)}")
                logger.warning(f"Raw content: {obj.properties['content']}")
                # Fallback: Construct a minimal customer object
                customer = {
                    "customer_id": obj.properties["title"],
                    "details": {"raw_content": obj.properties["content"][:500]}
                }
                customers.append(customer)
                logger.info(f"Added fallback customer profile for {obj.properties['title']}")

        # Filter by customer_id if provided
        if customer_id:
            customer = next((c for c in customers if c["customer_id"] == customer_id), None)
            if not customer:
                logger.warning(f"Customer {customer_id} not found")
                raise HTTPException(status_code=404, detail="Customer not found")
            logger.info(f"Successfully retrieved customer {customer_id}")
            return customer

        logger.info(f"Returning {len(customers)} customers")
        return {"customers": customers}
    except Exception as e:
        logger.error(f"Error fetching customers: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching customers: {str(e)}")

# Add a login route
@app.get("/login")
async def login(request: Request):
    logger.info("Initiating SSO login process")
    logger.debug(f"Current session: {dict(request.session)}")
    
    # Get the redirect URI (adjust based on your deployment)
    redirect_uri = str(request.url_for("auth_callback"))
    # For local development, keep http; for production, use https
    if request.headers.get("X-Forwarded-Proto") == "https":
        redirect_uri = redirect_uri.replace("http://", "https://")
    
    logger.info(f"Using redirect URI: {redirect_uri}")
    
    # Generate a unique state value for CSRF protection
    state = secrets.token_urlsafe(16)
    
    # Initiate the auth code flow (this replaces the deprecated method)
    flow = initiate_auth_flow(redirect_uri=redirect_uri, state=state)
    
    # Store the flow in session for later use in callback
    request.session["flow"] = flow
    request.session["state"] = state
    logger.debug(f"Generated state: {state}")
    logger.info(f"Auth flow initiated and stored in session")
    
    # Get the authorization URL from the flow
    auth_url = flow.get("auth_uri")
    if not auth_url:
        logger.error("Failed to generate authorization URL from flow")
        raise HTTPException(status_code=500, detail="Failed to generate authorization URL")
    
    logger.info(f"Redirecting to Azure AD for login")
    return RedirectResponse(auth_url)

# Add a redirect route to handle the response from Azure AD
@app.get("/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    logger.info("Handling callback from Azure AD")
    
    # Get query parameters from the URL
    query_params = dict(request.query_params)
    logger.debug(f"Callback query parameters: {list(query_params.keys())}")
    logger.debug(f"Current session: {dict(request.session)}")
    
    # Check for errors
    error = query_params.get("error")
    if error:
        logger.error(f"Authentication error: {error}")
        error_desc = query_params.get("error_description", "Unknown error")
        return templates.TemplateResponse(request, "login.html", {
            "error": f"Authentication failed: {error_desc}"
        })
    
    # Verify state for CSRF protection
    callback_state = query_params.get("state")
    session_state = request.session.get("state")
    logger.debug(f"State verification: session_state={session_state}, callback_state={callback_state}")
    if not callback_state or callback_state != session_state:
        logger.error("State mismatch - possible CSRF attack")
        return templates.TemplateResponse(request, "login.html", {
            "error": "Invalid state parameter. Please try logging in again."
        })
    
    # Get the stored flow from session
    flow = request.session.get("flow")
    if not flow:
        logger.error("Auth flow not found in session")
        return templates.TemplateResponse(request, "login.html", {
            "error": "Authentication flow expired. Please try logging in again."
        })
    
    # Check for authorization code
    code = query_params.get("code")
    if not code:
        logger.error("Authorization code not found in callback")
        return templates.TemplateResponse(request, "login.html", {
            "error": "Authorization code not found. Please try logging in again."
        })
    
    logger.info(f"Acquiring token using auth code flow")
    
    # Acquire token using the newer auth code flow method
    result = acquire_token_by_auth_code_flow(auth_response=query_params, flow=flow)
    
    logger.debug(f"Token acquisition result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
    
    if "access_token" in result:
        logger.info("Token acquired successfully")
        
        # Extract user information from ID token claims
        id_token_claims = result.get("id_token_claims", {})
        logger.debug(f"ID token claims: {id_token_claims}")
        logger.debug(f"Full token result keys: {list(result.keys())}")
        
        # Extract serializable account info, not the whole object
        account_info = {}
        if result.get("account"):
            account = result.get("account")
            account_info = {
                "home_account_id": account.get("home_account_id"),
                "environment": account.get("environment"),
                "username": account.get("username")
            }
        
        logger.debug(f"Serializable account info: {account_info}")

        # NOTE: Do NOT store the raw tokens in the session cookie, they are large and will
        # exceed the 4KB cookie size limit causing the session to be dropped.
        # We only persist the minimal user profile required by the app.
        user_info = {
            "account": account_info,  # Minimal account reference
            "name": id_token_claims.get("name") or id_token_claims.get("given_name", "User"),
            "email": id_token_claims.get("preferred_username") or id_token_claims.get("email", ""),
            "authenticated": True,
            "login_time": datetime.now().isoformat()
        }
        
        logger.info(f"Storing user info in session: name={user_info['name']}, email={user_info['email']}")
        logger.debug(f"Complete user info to store: {user_info}")
        
        # Store in session with explicit logging
        request.session["user"] = user_info
        logger.info(f"Session after storing user: {dict(request.session)}")
        
        # Verify the storage worked
        stored_user = request.session.get("user")
        logger.info(f"Verification - Retrieved user from session: {stored_user is not None}")
        if stored_user:
            logger.debug(f"Stored user details: {stored_user}")
        
        # Clear the flow and state from session
        request.session.pop("flow", None)
        request.session.pop("state", None)
        
        logger.info("Authentication successful, redirecting to home page")
        # Redirect to home page after successful login
        return RedirectResponse(url="/", status_code=302)
    else:
        logger.error("Failed to acquire token")
        error_msg = result.get("error", "Unknown error")
        error_desc = result.get("error_description", "Failed to acquire token")
        logger.error(f"Error: {error_msg}, Description: {error_desc}")
        
        return templates.TemplateResponse(request, "login.html", {
            "error": f"Failed to acquire token: {error_desc}"
        })

# Add a logout route
@app.get("/logout")
async def logout(request: Request):
    logger.info("User logout initiated")
    logger.debug(f"Current session before logout: {dict(request.session)}")
    
    # Import TENANT_ID for logout URL
    from msal_auth import TENANT_ID

    # Get user from session
    user = request.session.get("user")
    if user and user.get("account"):
        # The user['account'] is the serializable dict we stored
        stored_account_info = user["account"]
        logger.info(f"Attempting to remove account from MSAL cache: {stored_account_info.get('username')}")
        try:
            # Use get_token_from_cache to find the actual account object in the cache
            # based on the username we stored in the session.
            from msal_auth import get_token_from_cache, remove_account
            
            # This will find the account matching the username and return token info, including the account object
            token_result = get_token_from_cache(account=stored_account_info)
            
            if token_result and token_result.get("account"):
                account_to_remove_from_cache = token_result.get("account")
                # Verify it's the correct account before removing
                if account_to_remove_from_cache.get("home_account_id") == stored_account_info.get("home_account_id"):
                    remove_account(account_to_remove_from_cache)
                    logger.info("Account successfully removed from MSAL cache")
                else:
                    logger.warning("Found a cached account for the user, but home_account_id did not match. Not removing.")
            else:
                logger.warning("Could not find a corresponding account in MSAL cache to remove.")

        except Exception as e:
            logger.error(f"Error removing account from cache: {str(e)}")
    
    # Clear application session
    request.session.clear()
    logger.info("Session cleared")
    
    # Redirect to Azure AD logout
    logout_url = f"https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/logout"
    post_logout_redirect_uri = str(request.url_for("home"))
    if request.headers.get("X-Forwarded-Proto") == "https":
        post_logout_redirect_uri = post_logout_redirect_uri.replace("http://", "https://")
    
    logger.info(f"Redirecting to Azure AD logout with post_logout_redirect_uri: {post_logout_redirect_uri}")
    return RedirectResponse(
        url=f"{logout_url}?post_logout_redirect_uri={post_logout_redirect_uri}",
        status_code=302
    )

# Add a route to get current user info
@app.get("/api/me")
async def get_current_user(request: Request):
    user = request.session.get("user")
    logger.debug(f"API /me called, user authenticated: {user is not None}")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return {
        "name": user.get("name", "User"),
        "email": user.get("email", ""),
        "authenticated": True,
        "login_time": user.get("login_time")
    }