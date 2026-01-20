from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from .config import settings
from .ingest.indexer import DocumentIndexer
from .qa.retriever import DocumentRetriever
from .qa.llm import LLMClient
from .cv.checker import check_cv
from .teacher.pdf_manager import PDFManager
from .teacher.pdf_metadata import PDFMetadataManager
from .notification.deadline_parser import DeadlineParser
from .notification.student_parser import StudentEmailParser
from .notification.email_sender import EmailSender
from .notification.scheduler import NotificationScheduler


# logging & constants
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
TEACHER_ID = "admin"
TEACHER_PASSWORD = "admin123@"


# Global instances
indexer = None
retriever = None
llm_client = None
pdf_manager = None
pdf_metadata_manager = None
notification_scheduler = None


# Users file helpers
USERS_FILE = Path(__file__).parent.parent / "data" / "users.json"
USERS_FILE.parent.mkdir(exist_ok=True)

def load_users():
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def init_users():
    users = load_users()
    if not users:
        users = {
            "1211101529": {
                "password": "123abc@",
                "user_type": "student"
            }
        }
        save_users(users)
    return users


# FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global indexer, retriever, llm_client, pdf_manager, pdf_metadata_manager, notification_scheduler
    logger.info("Starting up Industrial Training Chatbot...")
    init_users()
    indexer = DocumentIndexer()
    retriever = DocumentRetriever(indexer)
    try:
        if settings.GROQ_API_KEY:
            llm_client = LLMClient(use_groq=True)
            logger.info("Using Groq (Llama)")
        elif settings.GOOGLE_API_KEY and settings.GOOGLE_API_KEY != "PUT_YOUR_GOOGLE_API_KEY_HERE":
            llm_client = LLMClient(use_google=True)
            logger.info("Using Google AI (Gemini)")
        else:
            llm_client = LLMClient(use_google=False)
            logger.info("Using OpenAI or local fallback")
    except Exception as e:
        logger.error(f"Error initializing LLM client: {str(e)}")
        llm_client = LLMClient(use_google=False)
        logger.info("Using local fallback due to initialization error")
    
    pdf_manager = PDFManager()
    pdf_metadata_manager = PDFMetadataManager()
    logger.info("Teacher PDF management initialized")
    
    notification_scheduler = NotificationScheduler()
    notification_scheduler.start()
    logger.info("Notification scheduler initialized and started")
    
    logger.info(f"Indexing chatbot documents from: {settings.PDF_FOLDER}")
    index_result = indexer.index_directory(settings.PDF_FOLDER, pdf_type="chatbot")
    logger.info(f"Indexing complete: {index_result}")
    
    yield
    
    if notification_scheduler:
        notification_scheduler.stop()
    logger.info("Shutting down...")


# FastAPI app init
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    language: str


class LoginRequest(BaseModel):
    user_id: str
    password: str


class RegisterRequest(BaseModel):
    user_id: str
    password: str
    user_type: str = "student"  # "student" or "teacher"


class LoginResponse(BaseModel):
    success: bool
    message: str
    user_id: str | None = None
    user_type: str | None = None


app = FastAPI(title="Industrial Training FIST Chatbot", lifespan=lifespan)

# CORS: allow file:// or any dev origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#LOGIN LOGIC
@app.post("/api/login", response_model=LoginResponse)
def login(req: LoginRequest):
    user_id = req.user_id.strip()
    password = req.password

    # --- Teacher login (ONLY admin) ---
    if user_id == TEACHER_ID and password == TEACHER_PASSWORD:
        return LoginResponse(
            success=True,
            message="Teacher login successful",
            user_id=user_id,
            user_type="teacher"
        )

    # --- Student login ---
    users = load_users()
    if user_id in users and users[user_id]["password"] == password:
        return LoginResponse(
            success=True,
            message="Student login successful",
            user_id=user_id,
            user_type="student"
        )

    return LoginResponse(
        success=False,
        message="Invalid user ID or password"
    )


# REGISTER (STUDENT ONLY)
@app.post("/api/register", response_model=LoginResponse)
def register(req: RegisterRequest):
    users = load_users()
    user_id = req.user_id.strip()

    if not user_id:
        return LoginResponse(success=False, message="User ID cannot be empty")

    if user_id == TEACHER_ID:
        return LoginResponse(success=False, message="This user ID is reserved")

    if user_id in users:
        return LoginResponse(success=False, message="User ID already exists")

    users[user_id] = {
        "password": req.password,
        "user_type": "student"
    }
    save_users(users)

    return LoginResponse(
        success=True,
        message="Student registration successful",
        user_id=user_id,
        user_type="student"
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "pdf_folder": settings.PDF_FOLDER,
        "default_language": settings.DEFAULT_LANGUAGE,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = (req.message or "").strip()
    # Force English responses for consistency
    lang = "en"
    
    # Check if we have the RAG components ready
    if not retriever or not llm_client:
        reply = "System is still initializing. Please wait a moment and try again."
        return ChatResponse(reply=reply, language=lang)
    
    # Handle empty messages
    if not text:
        reply = "Hi! I'm your Industrial Training assistant. You can start asking questions anytime."
        return ChatResponse(reply=reply, language=lang)
    
    # Check for farewell keywords
    farewell_map = {
        "en": ["bye", "goodbye", "thank you", "thanks"],
    }
    lowered = text.lower()
    if any(k in lowered for k in farewell_map.get(lang, [])):
        reply = "Thanks for chatting! If you have more questions, just ask anytime."
        return ChatResponse(reply=reply, language=lang)
    
    try:
        # Retrieve relevant chunks - increase k for better coverage
        chunks = retriever.retrieve_relevant_chunks(text, k=8)
        
        if not chunks:
            reply = "I couldn't find that in the Industrial Training documents. Please rephrase or ask another question."
            return ChatResponse(reply=reply, language=lang)
        
        # Format context
        context = retriever.format_context(chunks)
        
        # Generate response using LLM
        llm_result = llm_client.generate_response(text, context, lang)
        reply = llm_result.get('response', 'Sorry, I could not generate a response.')
        
        # If confidence is low, add a clarification
        confidence = llm_result.get('confidence', 0.0)
        if confidence < 0.3:
            reply += " Could you provide more specific details about what you're looking for?"
        
        return ChatResponse(reply=reply, language=lang)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        reply = "Sorry, I encountered an error while processing your question. Please try again."
        return ChatResponse(reply=reply, language=lang)


def detect_language(text: str) -> str:
    if not text:
        return "en"  # Default to English
    
    # Check for Chinese characters first
    if any('\u4e00' <= ch <= '\u9fff' for ch in text):
        return "zh"
    
    # Check for Malay tokens
    malay_tokens = ["yang", "dan", "atau", "tidak", "sila", "permohonan", "latihan", "industri", "boleh", "adalah", "untuk", "dengan", "dari", "pada", "akan", "telah", "sudah"]
    lowered = text.lower()
    malay_count = sum(1 for tok in malay_tokens if tok in lowered)
    
    # Only use Malay if there are multiple Malay tokens
    if malay_count >= 2:
        return "ms"
    
    # Default to English for everything else
    return "en"


@app.get("/api/status")
def get_status():
    """Get system status and indexing information"""
    if not indexer:
        return {"status": "initializing", "message": "System is starting up..."}
    
    stats = indexer.get_stats(pdf_type="chatbot")  # Status endpoint shows chatbot stats
    return {
        "status": "ready",
        "indexed_documents": stats.get('total_vectors', 0),
        "pdf_folder": settings.PDF_FOLDER,
        "has_api_key": bool(settings.GROQ_API_KEY or settings.GOOGLE_API_KEY or settings.OPENAI_API_KEY)
    }

@app.post("/api/reindex")
def reindex_documents(pdf_type: str = "chatbot"):
    """Manually trigger document reindexing for a specific PDF type"""
    if not indexer:
        return {"error": "System not ready"}
    
    if pdf_type not in ["chatbot", "submission", "notification"]:
        return {"error": "Invalid pdf_type. Must be: chatbot, submission, or notification"}
    
    try:
        target_dir = pdf_manager.get_directory(pdf_type)
        result = indexer.index_directory(str(target_dir), pdf_type=pdf_type)
        return {"success": True, "result": result, "pdf_type": pdf_type}
    except Exception as e:
        logger.error(f"Reindexing error: {str(e)}")
        return {"error": str(e)}

@app.post("/api/teacher/clear-index")
def clear_index(pdf_type: str = "chatbot"):
    """Clear FAISS index for a specific PDF type (teacher only)"""
    if not indexer:
        return {"error": "System not ready"}
    
    if pdf_type not in ["chatbot", "submission", "notification"]:
        return {"error": "Invalid pdf_type. Must be: chatbot, submission, or notification"}
    
    try:
        # Clear the index for this PDF type
        indexer.clear_index(pdf_type=pdf_type)
        logger.info(f"FAISS index cleared for {pdf_type} by teacher")
        return {
            "success": True,
            "message": f"FAISS index for {pdf_type} cleared successfully."
        }
    except Exception as e:
        logger.error(f"Clear index error: {str(e)}")
        return {"error": str(e)}

@app.post("/api/teacher/rebuild-faiss-index")
def rebuild_faiss_index(pdf_type: str):
    """Rebuild FAISS index for a specific PDF type (teacher only)"""
    if not indexer:
        return {"error": "System not ready"}
    
    if pdf_type not in ["chatbot", "submission", "notification"]:
        return {"error": "Invalid pdf_type. Must be: chatbot, submission, or notification"}
    
    try:
        # Get directory for this PDF type
        target_dir = pdf_manager.get_directory(pdf_type)
        
        # Clear old index for this type
        indexer.clear_index(pdf_type=pdf_type)
        logger.info(f"Cleared old index for {pdf_type}")
        
        # Reindex all PDFs in the directory
        result = indexer.index_directory(str(target_dir), pdf_type=pdf_type)
        logger.info(f"Reindexed {pdf_type} PDFs from: {target_dir}")
        
        # Update rebuild status for all PDFs in this type
        pdf_list = pdf_manager.list_pdfs(pdf_type)
        for pdf in pdf_list:
            pdf_metadata_manager.update_pdf_status(
                filename=pdf["file_name"],
                pdf_type=pdf_type,
                status_type="rebuild_status",
                status="success" if result.get("processed_files", 0) > 0 else "failed"
            )
        
        return {
            "success": True, 
            "message": f"Index for {pdf_type} rebuilt successfully",
            "result": result,
            "pdf_type": pdf_type
        }
    except Exception as e:
        logger.error(f"Rebuild index error: {str(e)}")
        return {"error": str(e)}


@app.post("/api/cv-check")
async def cv_check(file: UploadFile = File(...)):
    """CV checker endpoint"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        pdf_content = await file.read()
        result = check_cv(pdf_content)
        return result
    except Exception as e:
        logger.error(f"CV check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing CV: {str(e)}")


@app.post("/api/student/submit-cv")
async def student_submit_cv(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Student submits CV/Resume PDF to teacher.
    The file is stored in the same 'submission' PDF folder,
    but marked with the student's user_id as uploader.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    if not user_id:
        raise HTTPException(status_code=400, detail="Missing user_id")

    try:
        pdf_content = await file.read()

        # Save PDF in submission directory with uploader = student id
        result = pdf_manager.upload_pdf(
            file_content=pdf_content,
            filename=file.filename,
            pdf_type="submission",
            uploaded_by=user_id,
        )

        if not result.get("success"):
            raise HTTPException(
                status_code=500, detail=result.get("error", "Upload failed")
            )

        # Store metadata and mark upload status
        pdf_metadata_manager.add_pdf_metadata(
            filename=result["file_name"],
            pdf_type="submission",
            file_size=result["file_size"],
            uploaded_by=user_id,
        )
        pdf_metadata_manager.update_pdf_status(
            filename=result["file_name"],
            pdf_type="submission",
            status_type="upload_status",
            status="success",
        )

        return {
            "success": True,
            "file_name": result["file_name"],
            "pdf_type": "submission",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Student CV submit error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error submitting CV: {str(e)}")


def select_lang(options: dict[str, str], lang: str) -> str:
    return options.get(lang, options.get(settings.DEFAULT_LANGUAGE, next(iter(options.values()))))


# Helper function to check teacher permission
def check_teacher_permission(user_id: str = None) -> bool:
    """Check if user is a teacher"""
    if not user_id:
        return False
    users = load_users()
    return user_id in users and users[user_id].get("user_type") == "teacher"


# Teacher PDF Management API Endpoints
@app.post("/api/teacher/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    pdf_type: str = Form(...),
    user_id: str = Form(None)
):
    """Upload a PDF file (teacher only)"""
    # Note: In production, get user_id from session/token
    # For now, we'll accept it as a parameter or header
    
    if not pdf_type or pdf_type not in ["chatbot", "submission", "notification"]:
        raise HTTPException(status_code=400, detail="Invalid pdf_type. Must be: chatbot, submission, or notification")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        pdf_content = await file.read()
        uploaded_by = user_id or "teacher"
        
        result = pdf_manager.upload_pdf(
            file_content=pdf_content,
            filename=file.filename,
            pdf_type=pdf_type,
            uploaded_by=uploaded_by
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=500, detail=result.get("error", "Upload failed"))
        
        # Save metadata
        pdf_metadata_manager.add_pdf_metadata(
            filename=result["file_name"],
            pdf_type=pdf_type,
            file_size=result["file_size"],
            uploaded_by=uploaded_by
        )
        
        # Update upload status
        pdf_metadata_manager.update_pdf_status(
            filename=result["file_name"],
            pdf_type=pdf_type,
            status_type="upload_status",
            status="success"
        )
        
        # Note: Reindexing should be done manually via Rebuild button, not automatically on upload
        # This allows admin to upload multiple files and rebuild once
        
        return result
        
    except Exception as e:
        logger.error(f"PDF upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {str(e)}")


@app.get("/api/teacher/list-pdfs")
def list_pdfs(pdf_type: str):
    """List all PDFs of a specific type (teacher only)"""
    if pdf_type not in ["chatbot", "submission", "notification"]:
        raise HTTPException(status_code=400, detail="Invalid pdf_type. Must be: chatbot, submission, or notification")
    
    try:
        pdf_list = pdf_manager.list_pdfs(pdf_type)
        # Merge with metadata to include status information
        metadata_list = pdf_metadata_manager.list_pdf_metadata(pdf_type)
        metadata_dict = {item["file_name"]: item for item in metadata_list if "file_name" in item}
        
        # Merge file info with metadata
        for pdf in pdf_list:
            if pdf["file_name"] in metadata_dict:
                pdf["upload_status"] = metadata_dict[pdf["file_name"]].get("upload_status")
                pdf["rebuild_status"] = metadata_dict[pdf["file_name"]].get("rebuild_status")
                pdf["delete_status"] = metadata_dict[pdf["file_name"]].get("delete_status")
                # Status shows the most recent operation (delete > rebuild > upload)
                pdf["status"] = pdf.get("delete_status") or pdf.get("rebuild_status") or pdf.get("upload_status") or "pending"
                # Determine last action type (delete > rebuild > upload)
                if pdf.get("delete_status"):
                    pdf["last_action"] = "delete"
                elif pdf.get("rebuild_status"):
                    pdf["last_action"] = "rebuild"
                elif pdf.get("upload_status"):
                    pdf["last_action"] = "upload"
                else:
                    pdf["last_action"] = "none"
            else:
                pdf["status"] = "pending"
                pdf["last_action"] = "none"
        
        return {
            "success": True,
            "pdf_type": pdf_type,
            "files": pdf_list,
            "count": len(pdf_list)
        }
    except Exception as e:
        logger.error(f"List PDFs error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing PDFs: {str(e)}")


@app.get("/api/teacher/list-student-submissions")
def list_student_submissions():
    """
    List all student-submitted CV/Resume PDFs (submission type where uploader is not the teacher account).
    """
    pdf_type = "submission"
    try:
        pdf_list = pdf_manager.list_pdfs(pdf_type)
        metadata_list = pdf_metadata_manager.list_pdf_metadata(pdf_type)
        metadata_dict = {item["file_name"]: item for item in metadata_list if "file_name" in item}

        student_files = []
        for pdf in pdf_list:
            meta = metadata_dict.get(pdf["file_name"])
            # Treat anything not uploaded by the teacher admin as a student submission
            if meta and meta.get("uploaded_by") != TEACHER_ID:
                combined = pdf.copy()
                combined["uploaded_by"] = meta.get("uploaded_by")
                combined["upload_time"] = meta.get("upload_time", combined.get("upload_time"))
                student_files.append(combined)

        return {
            "success": True,
            "pdf_type": pdf_type,
            "files": student_files,
            "count": len(student_files),
        }
    except Exception as e:
        logger.error(f"List student submissions error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing student submissions: {str(e)}")


@app.delete("/api/teacher/delete-pdf")
def delete_pdf(filename: str, pdf_type: str):
    """Delete a PDF file (teacher only)"""
    if pdf_type not in ["chatbot", "submission", "notification"]:
        raise HTTPException(status_code=400, detail="Invalid pdf_type. Must be: chatbot, submission, or notification")
    
    try:
        result = pdf_manager.delete_pdf(filename, pdf_type)
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "File not found"))
        
        # Update delete status before removing metadata (note: file will be deleted, but we log it first if metadata exists)
        # Actually, since we're removing metadata, we should log the delete action before removal
        # But since the file is being deleted, we can't track it in metadata anymore
        # So we'll just proceed with deletion
        # Remove metadata (this also removes any status tracking)
        pdf_metadata_manager.remove_pdf_metadata(filename, pdf_type)
        
        # Note: Reindexing should be done manually via Rebuild button after deletion
        # This allows admin to delete multiple files and rebuild once
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete PDF error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")


@app.get("/api/teacher/pdf-info")
def get_pdf_info(filename: str, pdf_type: str):
    """Get information about a specific PDF (teacher only)"""
    if pdf_type not in ["chatbot", "submission", "notification"]:
        raise HTTPException(status_code=400, detail="Invalid pdf_type. Must be: chatbot, submission, or notification")
    
    try:
        pdf_info = pdf_manager.get_pdf_info(filename, pdf_type)
        if not pdf_info:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Add metadata if available
        metadata = pdf_metadata_manager.get_pdf_metadata(filename, pdf_type)
        if metadata:
            pdf_info["metadata"] = metadata
        
        return pdf_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get PDF info error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting PDF info: {str(e)}")


@app.get("/api/teacher/view-pdf")
def view_pdf(filename: str, pdf_type: str):
    """View/download a PDF file (teacher only)"""
    if pdf_type not in ["chatbot", "submission", "notification"]:
        raise HTTPException(status_code=400, detail="Invalid pdf_type. Must be: chatbot, submission, or notification")
    
    try:
        from fastapi.responses import FileResponse
        target_dir = pdf_manager.get_directory(pdf_type)
        file_path = target_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/pdf"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"View PDF error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error viewing PDF: {str(e)}")


# Notification API Endpoints
@app.post("/api/teacher/upload-emails")
async def upload_student_emails(file: UploadFile = File(...)):
    """Upload student email list (CSV or TXT)"""
    try:
        file_content = await file.read()
        student_parser = StudentEmailParser()
        result = student_parser.parse_email_file(file_content, file.filename, save_file=True)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to parse email file"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload emails error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading emails: {str(e)}")


@app.get("/api/teacher/list-email-files")
def list_email_files():
    """List all uploaded email files"""
    try:
        student_parser = StudentEmailParser()
        files = student_parser.list_uploaded_files()
        return {
            "success": True,
            "files": files,
            "count": len(files)
        }
    except Exception as e:
        logger.error(f"List email files error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing email files: {str(e)}")


@app.delete("/api/teacher/delete-email-file")
def delete_email_file(filename: str):
    """Delete an uploaded email file"""
    try:
        student_parser = StudentEmailParser()
        result = student_parser.delete_uploaded_file(filename)
        
        if not result.get("success"):
            raise HTTPException(status_code=404, detail=result.get("error", "File not found"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete email file error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting email file: {str(e)}")


@app.get("/api/teacher/view-email-file")
def view_email_file(filename: str):
    """View/download an uploaded email file"""
    try:
        from fastapi.responses import FileResponse
        student_parser = StudentEmailParser()
        file_path = student_parser.get_uploaded_file_path(filename)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine media type
        media_type = "text/plain"
        if filename.lower().endswith('.csv'):
            media_type = "text/csv"
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type=media_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"View email file error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error viewing email file: {str(e)}")


@app.post("/api/teacher/parse-deadline-pdf")
async def parse_deadline_pdf():
    """Parse deadline information from the latest notification PDF"""
    try:
        deadline_parser = DeadlineParser()
        result = deadline_parser.parse_deadline_pdf()
        
        if result.get("error"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        
        # Save deadline info
        if notification_scheduler:
            notification_scheduler.save_deadline_info(result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Parse deadline PDF error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing deadline PDF: {str(e)}")


@app.get("/api/teacher/notification-status")
def get_notification_status():
    """Get notification status and schedule"""
    if not notification_scheduler:
        return {"error": "Notification scheduler not initialized"}
    
    try:
        status = notification_scheduler.get_notification_status()
        return status
    except Exception as e:
        logger.error(f"Get notification status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting notification status: {str(e)}")


@app.post("/api/teacher/send-notification")
def send_notification_manual(reminder_type: str = "general"):
    """Manually trigger notification sending"""
    if not notification_scheduler:
        raise HTTPException(status_code=500, detail="Notification scheduler not initialized")
    
    try:
        result = notification_scheduler.manual_send_notification(reminder_type)
        return result
    except Exception as e:
        logger.error(f"Manual send notification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending notification: {str(e)}")


@app.get("/api/teacher/notification-history")
def get_notification_history(limit: int = 50):
    """Get notification history"""
    if not notification_scheduler:
        return {"history": []}
    
    try:
        history = notification_scheduler.get_notification_history(limit)
        return {
            "success": True,
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Get notification history error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting notification history: {str(e)}")


# --- Frontend static files (must be mounted AFTER /api routes) ---
PROJECT_ROOT = Path(__file__).parent.parent
WEB_DIR = PROJECT_ROOT / "web"
if WEB_DIR.exists():
    # Serve web/index.html at "/" and keep relative links working
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")




