from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
from logging.handlers import RotatingFileHandler
from pydantic import BaseModel
import io
import os
import tempfile
from datetime import datetime
from multilingual_voice_assistant import MultilingualVoiceAssistant
from pdf_uploader import PDFUploader
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import PyPDF2
import docx

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler("insurance_assistant.log", maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize assistant with error handling
try:
    assistant = MultilingualVoiceAssistant()
    logger.info("MultilingualVoiceAssistant initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MultilingualVoiceAssistant: {e}")
    raise Exception(f"Failed to initialize MultilingualVoiceAssistant: {e}")

# Initialize PDF uploader
WEAVIATE_CONFIG = {
    "http_host": "localhost",
    "http_port": 8080,
    "grpc_host": "localhost",
    "grpc_port": 50051,
    "secure": False
}
AZURE_CONFIG = {
    "api_key": os.getenv("AZURE_OPENAI_API_KEY", "7WYgbnwGu1U7OoVI2ZUdgAzp9zQ2uLCcpFR5xNCSBIYP9DcwLYRGJQQJ99BEACYeBjFXJ3w3AAABACOGKn3Y"),
    "endpoint": "https://idkrag.openai.azure.com/",
    "api_version": "2025-01-01-preview",
    "embedding_deployment": "text-embedding-ada-002",
    "gpt_deployment": "gpt-4o"
}
try:
    pdf_uploader = PDFUploader(WEAVIATE_CONFIG, AZURE_CONFIG)
    pdf_uploader.create_collection("InsuranceDocs", "Collection for storing insurance PDF contents")
    logger.info("PDFUploader initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize PDFUploader: {e}")
    raise Exception(f"Failed to initialize PDFUploader: {e}")

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

class ModeRequest(BaseModel):
    mode: str
    session_id: str

class LanguageRequest(BaseModel):
    language: str
    session_id: str

def escape_html(text):
    """Escape HTML characters for safe display."""
    if not text:
        return text
    replacements = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&apos;'
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text

def format_bold_tags(content):
    if not content:
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
    return parts

@app.get("/")
async def serve_home():
    try:
        file_path = "templates/home.html"
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} not found")
            raise HTTPException(status_code=404, detail="Home page not found")
        logger.info("Serving home.html")
        return FileResponse(file_path)
    except Exception as e:
        logger.error(f"Error serving home page: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/{page}.html")
async def serve_page(page: str):
    try:
        valid_pages = ["home", "chat", "settings", "conversations", "knowledge", "insights", "product_qa"]
        if page not in valid_pages:
            logger.warning(f"Invalid page requested: {page}")
            raise HTTPException(status_code=404, detail="Page not found")
        file_path = f"templates/{page}.html"
        if not os.path.exists(file_path):
            logger.error(f"File {file_path} not found")
            raise HTTPException(status_code=404, detail=f"File {page}.html not found")
        logger.info(f"Serving {page}.html")
        return FileResponse(file_path)
    except Exception as e:
        logger.error(f"Error serving page {page}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        if not request.session_id:
            logger.warning("Missing session_id in chat request")
            raise HTTPException(status_code=400, detail="Session ID is required")
        if not request.message.strip():
            logger.warning("Empty message in chat request")
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Detect Product Q&A session
        is_product_qa = request.session_id.startswith("product_qa_")
        context = "product_qa" if is_product_qa else "general"
        modified_message = f"[Context: {context}] {request.message}" if is_product_qa else request.message

        response, audio_data, suggestions = await assistant.process_user_input(
            text_input=modified_message,
            session_id=request.session_id,
            language=request.language,
            input_mode=request.input_mode,
            output_mode=request.output_mode
        )
        logger.info(f"Generated response for session {request.session_id} (context: {context}): {response}")
        logger.info(f"Generated suggestions: {suggestions}")
        return {
            "response": response,
            "audio_data": audio_data.hex() if isinstance(audio_data, bytes) else None,
            "suggestions": suggestions or []
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing chat request for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/api/chat_history")
async def get_chat_history(session_id: str = None):
    try:
        history = assistant.get_conversation_history()
        if session_id:
            if session_id not in history:
                logger.warning(f"Session {session_id} not found in chat history")
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            history = {session_id: history.get(session_id, {"messages": [], "summary": "", "entities": {}})}
        logger.info(f"Returning chat history with {len(history)} sessions")
        return {"sessions": history}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@app.post("/api/set_input_mode")
async def set_input_mode(request: ModeRequest):
    try:
        if not request.session_id:
            logger.warning("Missing session_id in set_input_mode request")
            raise HTTPException(status_code=400, detail="Session ID is required")
        success = assistant.set_input_mode(request.mode)
        if not success:
            logger.warning(f"Invalid input mode: {request.mode}")
            raise HTTPException(status_code=400, detail="Invalid input mode")
        logger.info(f"Set input mode to {request.mode} for session {request.session_id}")
        return {"status": "success"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error setting input mode for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error setting input mode: {str(e)}")

@app.post("/api/set_output_mode")
async def set_output_mode(request: ModeRequest):
    try:
        if not request.session_id:
            logger.warning("Missing session_id in set_output_mode request")
            raise HTTPException(status_code=400, detail="Session ID is required")
        success = assistant.set_output_mode(request.mode)
        if not success:
            logger.warning(f"Invalid output mode: {request.mode}")
            raise HTTPException(status_code=400, detail="Invalid output mode")
        logger.info(f"Set output mode to {request.mode} for session {request.session_id}")
        return {"status": "success"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error setting output mode for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error setting output mode: {str(e)}")

@app.post("/api/set_language")
async def set_language(request: LanguageRequest):
    try:
        if not request.session_id:
            logger.warning("Missing session_id in set_language request")
            raise HTTPException(status_code=400, detail="Session ID is required")
        success = assistant.set_language(request.language)
        if not success:
            logger.warning(f"Invalid language: {request.language}")
            raise HTTPException(status_code=400, detail="Invalid language")
        logger.info(f"Set language to {request.language} for session {request.session_id}")
        return {"status": "success"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error setting language for session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error setting language: {str(e)}")

@app.get("/api/download_chat")
async def download_chat(session_id: str):
    try:
        if not session_id:
            logger.warning("Missing session_id in download_chat request")
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        # Load conversation history
        history = assistant.get_conversation_history()
        session = history.get(session_id, {"messages": []})
        if not session:
            logger.warning(f"Session {session_id} not found for PDF download")
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        messages = session.get("messages", [])

        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=72, bottomMargin=72, leftMargin=72, rightMargin=72)
        elements = []

        # Define styles
        title_style = ParagraphStyle(
            name='Title',
            fontSize=16,
            leading=20,
            textColor=colors.navy,
            spaceAfter=12
        )
        user_style = ParagraphStyle(
            name='User',
            fontSize=12,
            leading=14,
            textColor=colors.darkblue,
            spaceBefore=6,
            spaceAfter=2
        )
        assistant_style = ParagraphStyle(
            name='Assistant',
            fontSize=12,
            leading=14,
            textColor=colors.blue,
            spaceBefore=6,
            spaceAfter=2
        )
        assistant_bold_style = ParagraphStyle(
            name='AssistantBold',
            fontSize=12,
            leading=14,
            textColor=colors.blue,
            fontName='Helvetica-Bold',
            spaceBefore=6,
            spaceAfter=2
        )
        timestamp_style = ParagraphStyle(
            name='Timestamp',
            fontSize=10,
            leading=12,
            textColor=colors.grey,
            spaceAfter=6
        )
        suggestion_style = ParagraphStyle(
            name='Suggestion',
            fontSize=10,
            leading=12,
            textColor=colors.darkcyan,
            leftIndent=20
        )

        # Add title
        title = "AI Insurance Agent Product Q&A Session" if session_id.startswith("product_qa_") else "AI Insurance Agent Chat Session"
        elements.append(Paragraph(title, title_style))
        elements.append(Spacer(1, 12))

        # Format messages
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

        # Build PDF
        if not elements:
            elements.append(Paragraph("No messages in this session.", user_style))
        doc.build(elements)
        pdf_data = buffer.getvalue()
        buffer.close()
        logger.info(f"PDF generated successfully for session {session_id}")

        # Return PDF as streaming response
        return StreamingResponse(
            io.BytesIO(pdf_data),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=chat_session_{session_id}.pdf"}
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error generating PDF for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

@app.post("/api/upload_document")
async def upload_document(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.pdf', '.doc', '.docx')):
            logger.warning(f"Invalid file type uploaded: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF and Word documents are supported")

        # Read file content
        content = ""
        file_bytes = await file.read()
        if file.filename.lower().endswith('.pdf'):
            # Try PDFUploader first
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_file_path = tmp_file.name
            try:
                result = pdf_uploader.add_pdf_to_collection("InsuranceDocs", tmp_file_path)
                content = result.get("content", "") if isinstance(result, dict) else ""
                logger.info(f"PDFUploader extracted {len(content)} characters from {file.filename}")
            except Exception as e:
                logger.warning(f"PDFUploader failed for {file.filename}: {e}")
                content = ""
            finally:
                os.unlink(tmp_file_path)

            # Fallback to PyPDF2 if content is empty
            if not content.strip():
                logger.info(f"Falling back to PyPDF2 for {file.filename}")
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
                for page in pdf_reader.pages:
                    extracted_text = page.extract_text() or ""
                    content += extracted_text
                logger.info(f"PyPDF2 extracted {len(content)} characters from {file.filename}")

            # If still empty, warn but proceed with metadata
            if not content.strip():
                logger.warning(f"No text extracted from {file.filename}. Storing metadata only.")
                content = f"Metadata for {file.filename}"  # Placeholder content
                # Store metadata in Weaviate
                embedding = pdf_uploader.get_text_embedding(file.filename)
                if not embedding:
                    logger.error(f"Failed to generate embedding for {file.filename} metadata")
                    raise HTTPException(status_code=500, detail="Failed to generate document embedding")
                collection = pdf_uploader.client.collections.get("InsuranceDocs")
                collection.data.insert({
                    "title": file.filename,
                    "content": content
                }, vector=embedding)
                return {
                    "message": f"Document '{file.filename}' uploaded successfully (no text extracted, metadata stored)",
                    "warning": "No text could be extracted from the PDF. It may be scanned or encrypted."
                }

        elif file.filename.lower().endswith(('.doc', '.docx')):
            doc = docx.Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                content += para.text + "\n"
            logger.info(f"Extracted {len(content)} characters from {file.filename}")

        if not content.strip():
            logger.warning(f"Document is empty or unreadable: {file.filename}")
            raise HTTPException(status_code=400, detail="Document is empty or unreadable")

        # Store Word document or PDF with content in Weaviate
        if file.filename.lower().endswith(('.doc', '.docx')) or (file.filename.lower().endswith('.pdf') and content.strip()):
            embedding = pdf_uploader.get_text_embedding(content)
            if not embedding:
                logger.error(f"Failed to generate embedding for {file.filename}")
                raise HTTPException(status_code=500, detail="Failed to generate document embedding")
            collection = pdf_uploader.client.collections.get("InsuranceDocs")
            collection.data.insert({
                "title": file.filename,
                "content": content
            }, vector=embedding)

        logger.info(f"Uploaded document: {file.filename}")
        return {"message": f"Document '{file.filename}' uploaded successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error uploading document {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/api/list_documents")
async def list_documents():
    try:
        documents = pdf_uploader.list_all_documents("InsuranceDocs")
        logger.info(f"Retrieved {len(documents)} documents from InsuranceDocs")
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.delete("/api/delete_document/{document_id}")
async def delete_document(document_id: str):
    try:
        collection = pdf_uploader.client.collections.get("InsuranceDocs")
        result = collection.data.delete_by_id(uuid=document_id)
        if result:
            logger.info(f"Successfully deleted document {document_id}")
            return {"status": "success", "message": f"Document {document_id} deleted successfully"}
        else:
            logger.warning(f"Document {document_id} not found")
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.post("/api/clear_history")
async def clear_history():
    try:
        assistant.clear_conversation_history()
        if os.path.exists("conversation_history.json"):
            os.remove("conversation_history.json")
            logger.info("Conversation history file deleted")
        logger.info("Conversation history cleared successfully")
        return {"message": "Conversation history cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")