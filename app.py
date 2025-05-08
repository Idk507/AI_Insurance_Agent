from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging
import asyncio
import io
from multilingual_voice_assistant import MultilingualVoiceAssistant, LANGUAGES

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

# Initialize FastAPI app
app = FastAPI(title="Multilingual Insurance Assistant")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for UI assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Initialize the assistant (single instance for all sessions)
assistant = MultilingualVoiceAssistant()

# Request models
class ChatRequest(BaseModel):
    message: str
    language: str | None = None
    input_mode: str | None = None
    output_mode: str | None = None
    session_id: str

class ModeRequest(BaseModel):
    mode: str
    session_id: str

class LanguageRequest(BaseModel):
    language: str
    session_id: str

# Response model
class ChatResponse(BaseModel):
    response: str
    language: str
    audio_data: str | None = None  # Base64 encoded audio or null

@app.get("/")
async def serve_index(request: Request):
    """
    Serve the index.html file at the root endpoint.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        TemplateResponse with index.html
    """
    logger.info("Serving index.html at /")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Handle chat requests, process user input, and return a response with optional audio stream.
    
    Args:
        request: ChatRequest containing the user's message, language, modes, and session ID
        
    Returns:
        StreamingResponse with JSON or audio/mpeg content
    """
    try:
        logger.info(f"Received chat request: {request.message}, session_id: {request.session_id}")
        
        # Update language if provided
        if request.language:
            if not assistant.set_language(request.language):
                logger.warning(f"Invalid language: {request.language}")
                raise HTTPException(status_code=400, detail="Unsupported language")
        
        # Update modes if provided
        if request.input_mode:
            if not assistant.set_input_mode(request.input_mode):
                logger.warning(f"Invalid input mode: {request.input_mode}")
                raise HTTPException(status_code=400, detail="Invalid input mode")
        
        if request.output_mode:
            if not assistant.set_output_mode(request.output_mode):
                logger.warning(f"Invalid output mode: {request.output_mode}")
                raise HTTPException(status_code=400, detail="Invalid output mode")
        
        # Process the input
        response_text, audio_bytes = await assistant.process_user_input(request.message, request.session_id)
        
        if not response_text:
            error_msg = "Failed to generate a response. Please try again."
            if assistant.user_language != "en":
                error_msg = await assistant.weaviate_manager.translate_text(error_msg, "en", assistant.user_language)
            logger.error("Empty response from process_user_input")
            raise HTTPException(status_code=500, detail=error_msg)
        
        # Prepare response
        if request.output_mode in ["voice", "both"] and audio_bytes:
            import base64
            audio_data = base64.b64encode(audio_bytes).decode('utf-8')
            return {
                "response": response_text,
                "language": assistant.display_language,
                "audio_data": audio_data
            }
        else:
            return {
                "response": response_text,
                "language": assistant.display_language,
                "audio_data": None
            }
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        error_msg = "An error occurred while processing your request. Please try again."
        if assistant.user_language != "en":
            error_msg = await assistant.weaviate_manager.translate_text(error_msg, "en", assistant.user_language)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/set_language")
async def set_language(request: LanguageRequest):
    """
    Set the language for a session.
    
    Args:
        request: LanguageRequest with the language and session ID
        
    Returns:
        Confirmation message
    """
    try:
        if not assistant.set_language(request.language):
            logger.warning(f"Invalid language: {request.language}")
            raise HTTPException(status_code=400, detail="Unsupported language")
        
        logger.info(f"Set language to {request.language} for session {request.session_id}")
        return {"message": f"Language set to {request.language}"}
    except Exception as e:
        logger.error(f"Error setting language: {e}")
        raise HTTPException(status_code=500, detail="Failed to set language")

@app.post("/api/set_input_mode")
async def set_input_mode(request: ModeRequest):
    """
    Set the input mode for a session.
    
    Args:
        request: ModeRequest with the mode and session ID
        
    Returns:
        Confirmation message
    """
    try:
        if not assistant.set_input_mode(request.mode):
            logger.warning(f"Invalid input mode: {request.mode}")
            raise HTTPException(status_code=400, detail="Invalid input mode")
        
        logger.info(f"Set input mode to {request.mode} for session {request.session_id}")
        return {"message": f"Input mode set to {request.mode}"}
    except Exception as e:
        logger.error(f"Error setting input mode: {e}")
        raise HTTPException(status_code=500, detail="Failed to set input mode")

@app.post("/api/set_output_mode")
async def set_output_mode(request: ModeRequest):
    """
    Set the output mode for a session.
    
    Args:
        request: ModeRequest with the mode and session ID
        
    Returns:
        Confirmation message
    """
    try:
        if not assistant.set_output_mode(request.mode):
            logger.warning(f"Invalid output mode: {request.mode}")
            raise HTTPException(status_code=400, detail="Invalid output mode")
        
        logger.info(f"Set output mode to {request.mode} for session {request.session_id}")
        return {"message": f"Output mode set to {request.mode}"}
    except Exception as e:
        logger.error(f"Error setting output mode: {e}")
        raise HTTPException(status_code=500, detail="Failed to set output mode")

@app.get("/api/chat_history")
async def get_chat_history(session_id: str):
    """
    Return the chat history for all sessions.
    
    Args:
        session_id: Session ID (used for context, but returns all sessions)
        
    Returns:
        Dictionary of sessions with summaries and messages
    """
    try:
        logger.info(f"Fetching chat history for session {session_id}")
        history = assistant.get_conversation_history()
        return {"sessions": history}
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")

@app.get("/api/languages")
async def get_languages():
    """
    Return the list of supported languages.
    
    Returns:
        List of supported languages in the user's preferred language
    """
    try:
        languages = await assistant.get_supported_languages()
        logger.info("Fetched supported languages")
        return {"languages": languages}
    except Exception as e:
        logger.error(f"Error fetching languages: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch supported languages")