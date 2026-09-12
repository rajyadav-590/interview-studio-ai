from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import logging
import os
import time
import uuid

from pypdf import PdfReader

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
)
from fastapi.responses import FileResponse

from fastapi.security import (
    HTTPBearer,
    HTTPAuthorizationCredentials,
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from jose import jwt, JWTError

from passlib.context import CryptContext

from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    String,
    Integer,
    DateTime,
    Text,
    ForeignKey,
    text,
)

from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    sessionmaker,
    Session,
    relationship,
)

from .llm import (
    LLMService,
    answer_guardrail,
)

from .llm.service import (
    fallback_evaluation,
)

from .llm.schemas import (
    ResumeAnalysisResult,
    JDAnalysisResult,
)

from .ml.analytics import analyze_answer, aggregate_readiness, get_readiness_model


# ================================================================
# Environment
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
VIDEO_DIR = BASE_DIR / "uploads" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(
    BASE_DIR.parent / ".env"
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'interviewer.db'}",
)

JWT_SECRET = os.getenv(
    "JWT_SECRET",
    "dev-secret-change-me",
)


# ================================================================
# Database
# ================================================================

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    }
    if DATABASE_URL.startswith("sqlite")
    else {},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
)


# ================================================================
# Security / Services
# ================================================================

pwd = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)

logger = logging.getLogger(__name__)

llm_service = LLMService()


# ================================================================
# Rate Limiting
# ================================================================

ANSWER_RATE_LIMIT = int(
    os.getenv(
        "ANSWER_RATE_LIMIT",
        "5",
    )
)

ANSWER_RATE_WINDOW_SECONDS = 60

answer_attempts: Dict[
    int,
    List[float],
] = {}


# ================================================================
# Database Models
# ================================================================

class Base(DeclarativeBase):
    pass


class User(Base):

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    email: Mapped[str] = mapped_column(
        String(160),
        unique=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(100)
    )

    password_hash: Mapped[str] = mapped_column(
        String(255)
    )

    interviews: Mapped[
        List["Interview"]
    ] = relationship(
        back_populates="user",
        cascade="all, delete",
    )

    resumes: Mapped[
        List["Resume"]
    ] = relationship(
        back_populates="user",
        cascade="all, delete",
    )


class Resume(Base):

    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        index=True,
    )

    filename: Mapped[str] = mapped_column(
        String(255)
    )

    extracted_text: Mapped[str] = mapped_column(
        Text
    )

    analysis_json: Mapped[str] = mapped_column(
        Text
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(
            timezone.utc
        ),
    )

    user: Mapped[User] = relationship(
        back_populates="resumes"
    )


class JobDescription(Base):

    __tablename__ = "job_descriptions"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        index=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        default="Job Description",
    )

    description_text: Mapped[str] = mapped_column(
        Text
    )

    analysis_json: Mapped[str] = mapped_column(
        Text
    )

    match_json: Mapped[str] = mapped_column(
        Text,
        default="{}",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(
            timezone.utc
        ),
    )

    user: Mapped[User] = relationship()


class Interview(Base):

    __tablename__ = "interviews"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id")
    )

    role: Mapped[str] = mapped_column(
        String(120)
    )

    experience: Mapped[str] = mapped_column(
        String(40)
    )

    interview_type: Mapped[str] = mapped_column(
        String(40)
    )

    difficulty: Mapped[str] = mapped_column(
        String(30)
    )

    current_difficulty: Mapped[str] = mapped_column(
        String(30),
        default="Medium",
    )

    interview_source: Mapped[str] = mapped_column(
        String(30),
        default="new",
    )

    domain: Mapped[str] = mapped_column(
        String(80),
        default="General Software Engineering",
    )

    current_question: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    current_question_source: Mapped[str] = mapped_column(
        String(40), default="gemini"
    )

    current_question_topic: Mapped[Optional[str]] = mapped_column(
        String(160), nullable=True
    )

    current_question_fallback_reason: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    last_generation_error: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )

    total_questions: Mapped[int] = mapped_column(
        Integer
    )

    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
    )

    score: Mapped[
        Optional[int]
    ] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(
            timezone.utc
        ),
    )

    user: Mapped[User] = relationship(
        back_populates="interviews"
    )

    answers: Mapped[
        List["Answer"]
    ] = relationship(
        back_populates="interview",
        cascade="all, delete",
    )


class Answer(Base):

    __tablename__ = "answers"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    interview_id: Mapped[str] = mapped_column(
        ForeignKey("interviews.id")
    )

    question: Mapped[str] = mapped_column(
        Text
    )

    answer: Mapped[str] = mapped_column(
        Text
    )

    evaluation_json: Mapped[str] = mapped_column(
        Text
    )

    question_source: Mapped[str] = mapped_column(
        String(40), default="gemini"
    )

    question_topic: Mapped[Optional[str]] = mapped_column(
        String(160), nullable=True
    )

    interview: Mapped[Interview] = relationship(
        back_populates="answers"
    )


class VideoRecording(Base):

    __tablename__ = "video_recordings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    interview_id: Mapped[str] = mapped_column(ForeignKey("interviews.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    question_number: Mapped[int] = mapped_column(Integer, default=1)
    filename: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(100), default="video/webm")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    interview: Mapped["Interview"] = relationship()
    user: Mapped["User"] = relationship()


# ================================================================
# Create Tables
# ================================================================

Base.metadata.create_all(
    engine
)


def _migrate_interview_columns() -> None:
    """Add interview context columns to existing SQLite/Postgres databases.

    create_all() does not alter an already-created table, so this lightweight
    migration keeps existing student installations compatible.
    """
    from sqlalchemy import inspect

    inspector = inspect(engine)
    columns = {col["name"] for col in inspector.get_columns("interviews")}
    additions = {
        "current_difficulty": "VARCHAR(30) DEFAULT 'Medium'",
        "interview_source": "VARCHAR(30) DEFAULT 'new'",
        "domain": "VARCHAR(80) DEFAULT 'General Software Engineering'",
        "current_question": "TEXT",
        "current_question_source": "VARCHAR(40) DEFAULT 'gemini'",
        "current_question_topic": "VARCHAR(160)",
        "current_question_fallback_reason": "TEXT",
        "last_generation_error": "TEXT",
    }
    with engine.begin() as conn:
        for name, ddl in additions.items():
            if name not in columns:
                conn.execute(text(f"ALTER TABLE interviews ADD COLUMN {name} {ddl}"))

        # Existing rows inherit their original selected difficulty.
        if "current_difficulty" not in columns:
            conn.execute(text("UPDATE interviews SET current_difficulty = difficulty WHERE current_difficulty IS NULL OR current_difficulty = ''"))

_migrate_interview_columns()


def _migrate_answer_columns() -> None:
    """Add provenance columns to answers without requiring Alembic."""
    from sqlalchemy import inspect

    inspector = inspect(engine)
    columns = {col["name"] for col in inspector.get_columns("answers")}
    additions = {
        "question_source": "VARCHAR(40) DEFAULT 'gemini'",
        "question_topic": "VARCHAR(160)",
    }
    with engine.begin() as conn:
        for name, ddl in additions.items():
            if name not in columns:
                conn.execute(text(f"ALTER TABLE answers ADD COLUMN {name} {ddl}"))


_migrate_answer_columns()


# ================================================================
# FastAPI
# ================================================================

app = FastAPI(
    title="Interview Studio API",
    version="1.0.0",
)


# ================================================================
# CORS
# ================================================================

FRONTEND_URL = os.getenv(
    "FRONTEND_URL",
    "http://localhost:5173",
)

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

if FRONTEND_URL and FRONTEND_URL not in allowed_origins:
    allowed_origins.append(FRONTEND_URL)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=(
        r"https?://"
        r"(localhost|127\.0\.0\.1)"
        r"(:\d+)?$"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# Database Dependency
# ================================================================

def db():

    session = SessionLocal()

    try:
        yield session

    finally:
        session.close()


# ================================================================
# Authentication
# ================================================================

bearer_scheme = HTTPBearer(
    auto_error=False
)


def token_for(
    user: User,
):

    return jwt.encode(
        {
            "sub": str(user.id),

            "exp": (
                datetime.now(
                    timezone.utc
                )
                + timedelta(days=7)
            ),
        },
        JWT_SECRET,
        algorithm="HS256",
    )


def current_user(
    credentials: Optional[
        HTTPAuthorizationCredentials
    ] = Depends(
        bearer_scheme
    ),

    session: Session = Depends(db),
):

    if not credentials:

        raise HTTPException(
            status_code=401,
            detail="Authentication required",
        )

    try:

        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET,
            algorithms=["HS256"],
        )

        user_id = int(
            payload["sub"]
        )

    except (
        JWTError,
        ValueError,
        KeyError,
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid or expired session",
        )

    user = session.get(
        User,
        user_id,
    )

    if not user:

        raise HTTPException(
            status_code=401,
            detail="User not found",
        )

    return user


# ================================================================
# Request Models
# ================================================================

class AuthInput(BaseModel):

    email: str

    password: str = Field(
        min_length=6
    )

    name: Optional[str] = None


class SetupInput(BaseModel):

    role: str

    experience: str

    interview_type: str

    difficulty: str

    interview_source: str = "new"
    domain: str = "General Software Engineering"

    total_questions: int = Field(
        ge=5,
        le=20,
    )


class AnswerInput(BaseModel):

    question: str

    answer: str = Field(
        min_length=2
    )


class JDInput(BaseModel):

    title: str = Field(
        default="Job Description",
        max_length=255,
    )

    description: str = Field(
        min_length=20,
        max_length=30000,
    )


# ================================================================
# Resume Helper
# ================================================================

def latest_resume_analysis(
    user_id: int,
    session: Session,
) -> Optional[Dict[str, Any]]:

    """
    Get the latest analyzed resume for the user.

    The resume analysis is stored as JSON in the
    resumes table. If no resume exists, return None.

    This allows the interview system to continue
    working normally for users who have not uploaded
    a resume.
    """

    resume = (
        session.query(Resume)
        .filter_by(
            user_id=user_id
        )
        .order_by(
            Resume.created_at.desc()
        )
        .first()
    )

    if not resume:
        return None

    try:

        analysis = json.loads(
            resume.analysis_json
        )

    except (
        json.JSONDecodeError,
        TypeError,
    ):

        logger.warning(
            "Could not parse resume analysis for user %s",
            user_id,
        )

        return None

    if not isinstance(
        analysis,
        dict,
    ):

        return None

    return analysis


# ================================================================
# Interview Helpers
# ================================================================

def _interview_turns(interview: Interview) -> List[Dict[str, Any]]:
    """Convert persisted answers into the compact context used by the interviewer."""
    turns: List[Dict[str, Any]] = []
    for answer in interview.answers:
        try:
            evaluation = json.loads(answer.evaluation_json)
        except (json.JSONDecodeError, TypeError):
            evaluation = {}
        turns.append({
            "question": answer.question,
            "answer": answer.answer,
            "overall_score": evaluation.get("overall_score", 0),
            "relevance": evaluation.get("relevance", 0),
            "technical_knowledge": evaluation.get("technical_knowledge", 0),
            "clarity": evaluation.get("clarity", 0),
            "communication": evaluation.get("communication", 0),
        })
    return turns


def _generate_and_store_question(
    interview: Interview,
    session: Session,
) -> Any:
    """Generate exactly one next question and persist it as session state.

    Persisting the current question prevents refreshes, retries, LLM failures,
    and double-clicks from changing the question unexpectedly.
    """
    turns = _interview_turns(interview)
    use_profile_context = interview.interview_source in {"candidate_profile", "job_specific"}

    resume_analysis = None
    jd_analysis = None
    jd_match = None
    if use_profile_context:
        resume_analysis = latest_resume_analysis(interview.user_id, session)
        latest_jd = (
            session.query(JobDescription)
            .filter_by(user_id=interview.user_id)
            .order_by(JobDescription.created_at.desc())
            .first()
        )
        if latest_jd:
            try:
                jd_analysis = json.loads(latest_jd.analysis_json)
            except (json.JSONDecodeError, TypeError):
                jd_analysis = None
            try:
                jd_match = json.loads(latest_jd.match_json)
            except (json.JSONDecodeError, TypeError):
                jd_match = None

    result = llm_service.next_question(
        interview.role,
        interview.experience,
        interview.interview_type,
        interview.current_difficulty or interview.difficulty,
        turns,
        resume_analysis=resume_analysis,
        jd_analysis=jd_analysis,
        jd_match=jd_match,
        domain=interview.domain,
        interview_source=interview.interview_source,
    )

    question = (result.question or "").strip()
    if not question:
        raise RuntimeError("The interview engine returned an empty question.")

    interview.current_question = question
    interview.current_question_source = result.source or "gemini"
    bank_item = None
    try:
        from .interview.question_bank import find_question
        bank_item = find_question(question)
    except Exception:
        bank_item = None
    interview.current_question_topic = (bank_item or {}).get("topic") if bank_item else None
    interview.current_question_fallback_reason = result.fallback_reason
    interview.last_generation_error = None
    session.flush()
    return result


def question_for(interview: Interview, index: int, session: Session):
    """Return the persisted current question; generate only when missing."""
    if interview.status != "active":
        return None
    if interview.current_question:
        return interview.current_question
    _generate_and_store_question(interview, session)
    session.commit()
    session.refresh(interview)
    return interview.current_question


def interview_out(
    interview,
    next_question=None,
):

    return {
        "id": interview.id,

        "role": interview.role,

        "experience": interview.experience,

        "interview_type": interview.interview_type,

        "difficulty": interview.difficulty,

        "current_difficulty": interview.current_difficulty or interview.difficulty,

        "interview_source": interview.interview_source or "new",

        "domain": interview.domain or "General Software Engineering",

        "total_questions": interview.total_questions,

        "answered": len(
            interview.answers
        ),

        "status": interview.status,

        "score": interview.score,

        "question_source": interview.current_question_source or "gemini",

        "question_topic": interview.current_question_topic,

        "question_fallback_reason": interview.current_question_fallback_reason,

        "generation_error": interview.last_generation_error,

        "created_at": (
            interview.created_at.isoformat()
        ),

        "next_question": next_question,
    }


# ================================================================
# Rate Limit
# ================================================================

def enforce_answer_rate_limit(
    user_id: int,
) -> None:

    now = time.monotonic()

    recent = [
        stamp
        for stamp in answer_attempts.get(
            user_id,
            [],
        )
        if (
            now - stamp
            < ANSWER_RATE_WINDOW_SECONDS
        )
    ]

    if len(recent) >= ANSWER_RATE_LIMIT:

        raise HTTPException(
            status_code=429,
            detail=(
                "Answer submission limit reached. "
                "Please wait a minute before trying again."
            ),
            headers={
                "Retry-After": str(
                    ANSWER_RATE_WINDOW_SECONDS
                )
            },
        )

    recent.append(now)

    answer_attempts[user_id] = recent


# ================================================================
# Video Interview Routes
# ================================================================

@app.post("/api/interviews/{interview_id}/video")
async def upload_interview_video(
    interview_id: str,
    question_number: int = 1,
    duration_seconds: Optional[int] = None,
    video: UploadFile = File(...),
    user: User = Depends(current_user),
    session: Session = Depends(db),
):
    interview = session.get(Interview, interview_id)
    if not interview or interview.user_id != user.id:
        raise HTTPException(status_code=404, detail="Interview not found.")

    allowed = {"video/webm", "video/mp4", "video/quicktime", "video/x-matroska"}
    mime = (video.content_type or "").lower().split(";")[0].strip()
    filename_lower = (video.filename or "").lower()

    # Browsers sometimes send codec-specific WebM MIME types (for example
    # video/webm;codecs=vp9,opus) or omit the MIME type. Accept those when
    # the uploaded filename is clearly WebM/MP4.
    is_webm = mime == "video/webm" or "webm" in mime or filename_lower.endswith(".webm")
    is_mp4 = mime == "video/mp4" or "mp4" in mime or filename_lower.endswith(".mp4")
    is_mov = mime == "video/quicktime" or filename_lower.endswith(".mov")
    is_mkv = mime == "video/x-matroska" or filename_lower.endswith(".mkv")

    if mime and mime not in allowed and not (is_webm or is_mp4 or is_mov or is_mkv):
        raise HTTPException(status_code=400, detail="Unsupported video format. Use WebM or MP4.")

    data = await video.read()
    max_bytes = 50 * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="Video must be 50 MB or smaller.")

    extension = ".webm"
    if "mp4" in mime:
        extension = ".mp4"
    elif "quicktime" in mime:
        extension = ".mov"
    elif "matroska" in mime:
        extension = ".mkv"

    recording_id = str(uuid.uuid4())
    filename = f"{recording_id}{extension}"
    path = VIDEO_DIR / filename
    path.write_bytes(data)

    record = VideoRecording(
        id=recording_id,
        interview_id=interview_id,
        user_id=user.id,
        question_number=max(1, question_number),
        filename=filename,
        mime_type=mime,
        size_bytes=len(data),
        duration_seconds=duration_seconds,
    )
    session.add(record)
    session.commit()

    return {
        "id": record.id,
        "interview_id": interview_id,
        "question_number": record.question_number,
        "mime_type": record.mime_type,
        "size_bytes": record.size_bytes,
        "duration_seconds": record.duration_seconds,
        "url": f"/api/video/{record.id}",
    }


@app.get("/api/interviews/{interview_id}/videos")
def list_interview_videos(
    interview_id: str,
    user: User = Depends(current_user),
    session: Session = Depends(db),
):
    interview = session.get(Interview, interview_id)
    if not interview or interview.user_id != user.id:
        raise HTTPException(status_code=404, detail="Interview not found.")

    rows = (
        session.query(VideoRecording)
        .filter_by(interview_id=interview_id, user_id=user.id)
        .order_by(VideoRecording.question_number.asc(), VideoRecording.created_at.asc())
        .all()
    )
    return [
        {
            "id": row.id,
            "question_number": row.question_number,
            "mime_type": row.mime_type,
            "size_bytes": row.size_bytes,
            "duration_seconds": row.duration_seconds,
            "created_at": row.created_at.isoformat(),
            "url": f"/api/video/{row.id}",
        }
        for row in rows
    ]


@app.get("/api/video/{video_id}")
def get_interview_video(
    video_id: str,
    user: User = Depends(current_user),
    session: Session = Depends(db),
):
    row = session.get(VideoRecording, video_id)
    if not row or row.user_id != user.id:
        raise HTTPException(status_code=404, detail="Video not found.")

    path = VIDEO_DIR / row.filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file is missing.")

    return FileResponse(path, media_type=row.mime_type, filename=row.filename)


# ================================================================
# Health
# ================================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",

        "mode": llm_service.mode,

        "llm_configured": (
            llm_service.configured
        ),

        "model": (
            llm_service.model
            if llm_service.configured
            else None
        ),
    }


# ================================================================
# Authentication Routes
# ================================================================

@app.post("/api/auth/register")
def register(
    data: AuthInput,
    session: Session = Depends(db),
):

    if session.query(User).filter_by(
        email=data.email.lower()
    ).first():

        raise HTTPException(
            status_code=409,
            detail=(
                "An account with that email "
                "already exists"
            ),
        )

    user = User(
        email=data.email.lower(),

        name=(
            data.name
            or data.email.split("@")[0].title()
        ),

        password_hash=pwd.hash(
            data.password
        ),
    )

    session.add(user)

    session.commit()

    session.refresh(user)

    return {
        "token": token_for(user),

        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        },
    }


@app.post("/api/auth/login")
def login(
    data: AuthInput,
    session: Session = Depends(db),
):

    user = session.query(User).filter_by(
        email=data.email.lower()
    ).first()

    if (
        not user
        or not pwd.verify(
            data.password,
            user.password_hash,
        )
    ):

        raise HTTPException(
            status_code=401,
            detail=(
                "Email or password is incorrect"
            ),
        )

    return {
        "token": token_for(user),

        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
        },
    }


@app.get("/api/user/profile")
def profile(
    user: User = Depends(
        current_user
    ),
):

    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
    }


# ================================================================
# Resume Upload
# ================================================================

@app.post("/api/resume/upload")
async def upload_resume(
    file: UploadFile = File(...),

    user: User = Depends(
        current_user
    ),
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please select a resume file."
            ),
        )

    filename = file.filename.lower()

    if not filename.endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF resumes are supported "
                "right now."
            ),
        )

    try:

        file_bytes = await file.read()

        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded PDF is empty."
                ),
            )

        from io import BytesIO

        reader = PdfReader(
            BytesIO(file_bytes)
        )

        if not reader.pages:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The PDF does not contain "
                    "any pages."
                ),
            )

        extracted_pages = []

        for page in reader.pages:

            text = (
                page.extract_text()
                or ""
            )

            if text.strip():

                extracted_pages.append(
                    text.strip()
                )

        resume_text = (
            "\n\n".join(
                extracted_pages
            ).strip()
        )

        if not resume_text:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable text was found "
                    "in this PDF. "
                    "If this is a scanned/image-only "
                    "resume, OCR support will be "
                    "added later."
                ),
            )

        resume_text = resume_text[:30000]

        return {
            "message": (
                "Resume uploaded and text "
                "extracted successfully."
            ),

            "filename": file.filename,

            "pages": len(
                reader.pages
            ),

            "characters": len(
                resume_text
            ),

            "text": resume_text,
        }

    except HTTPException:

        raise

    except Exception as exc:

        logger.exception(
            "Resume extraction failed"
        )

        raise HTTPException(
            status_code=422,
            detail=(
                "Could not read the PDF resume: "
                f"{str(exc)}"
            ),
        )


# ================================================================
# AI Resume Analysis + Database Storage
# ================================================================

@app.post("/api/resume/analyze")
async def analyze_resume(
    file: UploadFile = File(...),

    user: User = Depends(
        current_user
    ),

    session: Session = Depends(db),
):

    """
    Upload a PDF resume, extract its text,
    analyze it using Gemini, and save the
    analysis to MySQL.

    Flow:

        PDF
         ↓
        Text extraction
         ↓
        Gemini analysis
         ↓
        ResumeAnalysisResult
         ↓
        MySQL
    """

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please select a resume file."
            ),
        )

    filename = file.filename.lower()

    if not filename.endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail=(
                "Only PDF resumes are supported "
                "right now."
            ),
        )

    if not llm_service.configured:

        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini AI is not configured. "
                "Please configure the Gemini API "
                "in the backend environment."
            ),
        )

    try:

        # --------------------------------------------------------
        # Read PDF
        # --------------------------------------------------------

        file_bytes = await file.read()

        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The uploaded PDF is empty."
                ),
            )

        # --------------------------------------------------------
        # Extract PDF text
        # --------------------------------------------------------

        from io import BytesIO

        reader = PdfReader(
            BytesIO(file_bytes)
        )

        if not reader.pages:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The PDF does not contain "
                    "any pages."
                ),
            )

        extracted_pages = []

        for page in reader.pages:

            text = (
                page.extract_text()
                or ""
            )

            if text.strip():

                extracted_pages.append(
                    text.strip()
                )

        resume_text = (
            "\n\n".join(
                extracted_pages
            ).strip()
        )

        if not resume_text:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No readable text was found "
                    "in this PDF. "
                    "If this is a scanned/image-only "
                    "resume, OCR support will be "
                    "added later."
                ),
            )

        # --------------------------------------------------------
        # Limit resume size
        # --------------------------------------------------------

        resume_text = resume_text[:30000]

        # --------------------------------------------------------
        # Gemini Analysis
        # --------------------------------------------------------

        analysis = (
            llm_service.analyze_resume(
                resume_text
            )
        )

        if analysis is None:

            raise HTTPException(
                status_code=502,
                detail=(
                    "Gemini could not analyze "
                    "the resume. Please try again."
                ),
            )

        # --------------------------------------------------------
        # Convert analysis to dictionary
        # --------------------------------------------------------

        if isinstance(
            analysis,
            ResumeAnalysisResult,
        ):

            analysis_payload = (
                analysis.model_dump()
            )

        else:

            analysis_payload = analysis

        # --------------------------------------------------------
        # Save Resume Analysis
        # --------------------------------------------------------

        resume_record = Resume(

            user_id=user.id,

            filename=file.filename,

            extracted_text=resume_text,

            analysis_json=json.dumps(
                analysis_payload,
                ensure_ascii=False,
            ),
        )

        session.add(
            resume_record
        )

        session.commit()

        session.refresh(
            resume_record
        )

        # --------------------------------------------------------
        # Return Result
        # --------------------------------------------------------

        return {

            "message": (
                "Resume uploaded, extracted, "
                "analyzed, and saved successfully."
            ),

            "resume_id": (
                resume_record.id
            ),

            "filename": file.filename,

            "pages": len(
                reader.pages
            ),

            "characters": len(
                resume_text
            ),

            "analysis_mode": (
            "local" if "Local evidence-based analysis" in (analysis_payload.get("resume_quality_assessment") or "") else "gemini"
        ),

            "analysis": analysis_payload,
        }

    except HTTPException:

        raise

    except Exception:

        session.rollback()

        logger.exception(
            "AI resume analysis failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "An unexpected error occurred "
                "while analyzing and saving "
                "the resume."
            ),
        )


# ================================================================
# Get Latest Resume Analysis
# ================================================================

@app.get("/api/resume/latest")
def get_latest_resume(
    user: User = Depends(
        current_user
    ),

    session: Session = Depends(db),
):

    resume = (
        session.query(Resume)
        .filter_by(
            user_id=user.id
        )
        .order_by(
            Resume.created_at.desc()
        )
        .first()
    )

    if not resume:

        raise HTTPException(
            status_code=404,
            detail=(
                "No analyzed resume found. "
                "Please upload a resume first."
            ),
        )

    try:

        analysis = json.loads(
            resume.analysis_json
        )

    except (
        json.JSONDecodeError,
        TypeError,
    ):

        analysis = {}

    return {

        "resume_id": resume.id,

        "filename": resume.filename,

        "created_at": (
            resume.created_at.isoformat()
        ),

        "characters": len(
            resume.extracted_text
        ),

        "analysis": analysis,
    }



# ================================================================
# Job Description Analysis
# ================================================================

@app.post("/api/job-description/analyze")
def analyze_job_description(
    data: JDInput,

    user: User = Depends(
        current_user
    ),

    session: Session = Depends(db),
):

    if not llm_service.configured:

        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini AI is not configured."
            ),
        )

    jd_text = data.description.strip()

    if len(jd_text) < 20:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please provide a meaningful "
                "job description."
            ),
        )

    analysis = (
        llm_service.analyze_job_description(
            jd_text
        )
    )

    if analysis is None:

        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini could not analyze "
                "the job description."
            ),
        )

    if isinstance(
        analysis,
        JDAnalysisResult,
    ):

        analysis_payload = (
            analysis.model_dump()
        )

    else:

        analysis_payload = analysis

    # ------------------------------------------------------------
    # Resume ↔ JD Matching
    # ------------------------------------------------------------

    resume_analysis = latest_resume_analysis(
        user.id,
        session,
    )

    match_payload = {}

    if resume_analysis:

        match_result = (
            llm_service.match_resume_to_job(
                resume_analysis,
                analysis_payload,
            )
        )

        if match_result is not None:

            match_payload = (
                match_result.model_dump()
            )

    # ------------------------------------------------------------
    # Save JD
    # ------------------------------------------------------------

    jd_record = JobDescription(

        user_id=user.id,

        title=(
            data.title.strip()
            or "Job Description"
        ),

        description_text=jd_text,

        analysis_json=json.dumps(
            analysis_payload,
            ensure_ascii=False,
        ),

        match_json=json.dumps(
            match_payload,
            ensure_ascii=False,
        ),
    )

    session.add(
        jd_record
    )

    session.commit()

    session.refresh(
        jd_record
    )

    return {

        "message": (
            "Job description analyzed "
            "and saved successfully."
        ),

        "job_description_id": (
            jd_record.id
        ),

        "analysis_mode": (
            "local" if "Local evidence-based extraction" in json.dumps(analysis_payload, ensure_ascii=False) else "gemini"
        ),

        "analysis": analysis_payload,

        "resume_match": (
            match_payload
            if match_payload
            else None
        ),
    }


# ================================================================
# Get Latest Job Description
# ================================================================

@app.get("/api/job-description/latest")
def get_latest_job_description(
    user: User = Depends(
        current_user
    ),

    session: Session = Depends(db),
):

    jd = (
        session.query(JobDescription)
        .filter_by(
            user_id=user.id
        )
        .order_by(
            JobDescription.created_at.desc()
        )
        .first()
    )

    if not jd:

        raise HTTPException(
            status_code=404,
            detail=(
                "No job description found."
            ),
        )

    try:

        analysis = json.loads(
            jd.analysis_json
        )

    except (
        json.JSONDecodeError,
        TypeError,
    ):

        analysis = {}

    try:

        resume_match = json.loads(
            jd.match_json
        )

    except (
        json.JSONDecodeError,
        TypeError,
    ):

        resume_match = {}

    return {

        "job_description_id": jd.id,

        "title": jd.title,

        "description": (
            jd.description_text
        ),

        "created_at": (
            jd.created_at.isoformat()
        ),

        "analysis": analysis,

        "resume_match": (
            resume_match
            if resume_match
            else None
        ),
    }


# ================================================================
# Recalculate Resume ↔ JD Match
# ================================================================

@app.post(
    "/api/job-description/{job_description_id}/match"
)
def match_job_description(
    job_description_id: int,

    user: User = Depends(
        current_user
    ),

    session: Session = Depends(db),
):

    jd = session.get(
        JobDescription,
        job_description_id,
    )

    if (
        not jd
        or jd.user_id != user.id
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Job description not found."
            ),
        )

    resume_analysis = latest_resume_analysis(
        user.id,
        session,
    )

    if not resume_analysis:

        raise HTTPException(
            status_code=404,
            detail=(
                "Please upload and analyze "
                "a resume before matching it "
                "with a job description."
            ),
        )

    try:

        jd_analysis = json.loads(
            jd.analysis_json
        )

    except (
        json.JSONDecodeError,
        TypeError,
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "Saved job description analysis "
                "could not be read."
            ),
        )

    match_result = (
        llm_service.match_resume_to_job(
            resume_analysis,
            jd_analysis,
        )
    )

    if match_result is None:

        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini could not calculate "
                "the resume-job match."
            ),
        )

    match_payload = (
        match_result.model_dump()
    )

    jd.match_json = json.dumps(
        match_payload,
        ensure_ascii=False,
    )

    session.commit()

    return {

        "job_description_id": jd.id,

        "resume_match": match_payload,
    }


# ================================================================
# Create Interview
# ================================================================

@app.post("/api/interviews")
def create_interview(
    data: SetupInput,

    user: User = Depends(
        current_user
    ),

    session: Session = Depends(db),
):

    interview = Interview(

        id=str(
            uuid.uuid4()
        ),

        user_id=user.id,

        role=data.role,

        experience=data.experience,

        interview_type=data.interview_type,

        difficulty=data.difficulty,
        current_difficulty=data.difficulty,
        interview_source=data.interview_source,
        domain=data.domain,

        total_questions=data.total_questions,
    )

    session.add(
        interview
    )

    session.commit()

    session.refresh(
        interview
    )

    return interview_out(
        interview,

        question_for(
            interview,
            0,
            session,
        ),
    )


# ================================================================
# List Interviews
# ================================================================

@app.get("/api/interviews")
def list_interviews(
    user: User = Depends(
        current_user
    ),

    session: Session = Depends(db),
):

    items = (
        session.query(Interview)
        .filter_by(
            user_id=user.id
        )
        .order_by(
            Interview.created_at.desc()
        )
        .all()
    )

    completed = [
        x.score
        for x in items
        if x.score is not None
    ]

    return {

        "items": [
            interview_out(x)
            for x in items
        ],

        "stats": {

            "completed": len(
                completed
            ),

            "average": (
                round(
                    sum(completed)
                    / len(completed)
                )
                if completed
                else 0
            ),

            "best": (
                max(completed)
                if completed
                else 0
            ),
        },
    }


# ================================================================
# Get Interview
# ================================================================

@app.get(
    "/api/interviews/{interview_id}"
)
def get_interview(
    interview_id: str,

    user: User = Depends(
        current_user
    ),

    session: Session = Depends(db),
):

    interview = session.get(
        Interview,
        interview_id,
    )

    if (
        not interview
        or interview.user_id != user.id
    ):

        raise HTTPException(
            status_code=404,
            detail="Interview not found",
        )

    next_question = None

    if interview.status == "active":

        next_question = question_for(
            interview,
            len(
                interview.answers
            ),
            session,
        )

    return interview_out(
        interview,
        next_question,
    )


# ================================================================
# Submit Answer
# ================================================================

@app.post(
    "/api/interviews/{interview_id}/answer"
)
def submit_answer(
    interview_id: str,

    data: AnswerInput,

    user: User = Depends(
        current_user
    ),

    session: Session = Depends(db),
):

    interview = session.get(
        Interview,
        interview_id,
    )

    if (
        not interview
        or interview.user_id != user.id
    ):

        raise HTTPException(
            status_code=404,
            detail="Interview not found",
        )

    if interview.status != "active":

        raise HTTPException(
            status_code=400,
            detail=(
                "This interview is already complete"
            ),
        )

    # Idempotency guard: a browser retry, double-click, or network timeout
    # must never evaluate/save the same interview question twice.
    submitted_question = (data.question or "").strip()

    for existing in interview.answers:
        if (existing.question or "").strip().lower() == submitted_question.lower():
            try:
                existing_eval = json.loads(existing.evaluation_json)
            except (json.JSONDecodeError, TypeError):
                existing_eval = {}
            next_question = None
            if interview.status == "active":
                next_question = question_for(interview, len(interview.answers), session)
            return {
                "evaluation": existing_eval,
                "evaluation_mode": existing_eval.get("source", "local"),
                "llm_mode": llm_service.mode,
                "duplicate_submission": True,
                "interview": interview_out(interview, next_question),
            }

    # The browser must submit the exact active question. This prevents a stale
    # UI state from attaching a new answer to a different server-side question.
    if interview.current_question and submitted_question.lower() != interview.current_question.strip().lower():
        raise HTTPException(
            status_code=409,
            detail="This question is no longer the active interview question. Refresh the interview to continue.",
        )

    enforce_answer_rate_limit(
        user.id
    )

    # ------------------------------------------------------------
    # Answer Guardrail
    # ------------------------------------------------------------

    is_valid, guardrail_reason = (
        answer_guardrail(
            data.answer
        )
    )

    if not is_valid:

        result = fallback_evaluation(
            guardrail_reason,
            source="guardrail",
        )

    else:

        result = llm_service.evaluate(
            data.question,
            data.answer,
            interview.role,
            interview.experience,
            interview.current_difficulty or interview.difficulty,
            interview.interview_type,
        )

    result_payload = (
        result.model_dump()
    )

    # ------------------------------------------------------------
    # Adaptive difficulty controller
    # ------------------------------------------------------------
    # Difficulty is a session state, not merely the user's initial choice.
    # Strong answers increase challenge; weak answers reduce it; middle scores
    # keep the current level stable. Gemini receives the updated level on the
    # next question request.
    score = float(result_payload.get("overall_score", 0) or 0)
    current = (interview.current_difficulty or interview.difficulty or "Medium").title()
    levels = ["Easy", "Medium", "Hard"]
    try:
        idx = levels.index(current)
    except ValueError:
        idx = 1
    if score >= 8:
        idx = min(2, idx + 1)
    elif score < 5:
        idx = max(0, idx - 1)
    interview.current_difficulty = levels[idx]

    # ------------------------------------------------------------
    # Save Answer
    # ------------------------------------------------------------

    answer_record = Answer(

        interview_id=interview.id,

        question=data.question,

        answer=data.answer,

        evaluation_json=json.dumps(
            result_payload
        ),

        question_source=interview.current_question_source or "gemini",

        question_topic=interview.current_question_topic,
    )

    session.add(
        answer_record
    )

    session.commit()

    session.refresh(
        interview
    )

    # ------------------------------------------------------------
    # Complete Interview
    # ------------------------------------------------------------

    if (
        len(interview.answers)
        >= interview.total_questions
    ):

        interview.status = "completed"

        scores = []

        for answer in interview.answers:

            try:

                evaluation = json.loads(
                    answer.evaluation_json
                )

                scores.append(
                    float(
                        evaluation.get(
                            "overall_score",
                            0,
                        )
                    )
                )

            except (
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):

                scores.append(0)

        if scores:

            interview.score = round(
                (
                    sum(scores)
                    / len(scores)
                )
                * 10
            )

        else:

            interview.score = 0

        session.commit()

    # ------------------------------------------------------------
    # Next Question
    # ------------------------------------------------------------

    next_question = None
    generation_warning = None

    if interview.status == "active":
        # The previous question is now answered. Generate and persist exactly
        # one new question. If an unexpected engine error occurs, recover from
        # the curated bank instead of returning HTTP 500 and leaving the browser
        # stuck on the previous question.
        interview.current_question = None
        interview.current_question_source = "gemini"
        interview.current_question_topic = None
        interview.current_question_fallback_reason = None
        try:
            result = _generate_and_store_question(interview, session)
            next_question = result.question
        except Exception as exc:
            logger.exception("Question generation failed after answer; using emergency fallback: %s", exc)
            try:
                from .interview.question_bank import questions_for_domain, normalize_question
                used = {normalize_question(a.question) for a in interview.answers}
                candidates = questions_for_domain(interview.domain)
                emergency = next((item for item in candidates if normalize_question(item["question"]) not in used), None)
                if emergency is None:
                    candidates = questions_for_domain("General Software Engineering")
                    emergency = next((item for item in candidates if normalize_question(item["question"]) not in used), None)
                if emergency is None:
                    raise RuntimeError("No unused fallback question is available.")
                interview.current_question = emergency["question"]
                interview.current_question_source = "question-bank"
                interview.current_question_topic = emergency.get("topic")
                interview.current_question_fallback_reason = "Emergency local recovery was used after an unexpected question-generation error."
                interview.last_generation_error = str(exc)[:500]
                next_question = interview.current_question
                generation_warning = "AI question generation was temporarily unavailable. The interview continued with a unique fallback question."
            except Exception as emergency_exc:
                interview.last_generation_error = str(emergency_exc)[:500]
                generation_warning = "The next question could not be generated. Refresh and retry; your submitted answer has already been saved."

        session.commit()
        session.refresh(interview)

    return {

        "evaluation": result_payload,

        "evaluation_mode": (
            result.source
        ),

        "llm_mode": (
            llm_service.mode
        ),

        "generation_warning": generation_warning,

        "interview": interview_out(
            interview,
            next_question,
        ),
    }


# ================================================================
# Final AI Report
# ================================================================

@app.get(
    "/api/interviews/{interview_id}/report"
)
def report(
    interview_id: str,

    user: User = Depends(
        current_user
    ),

    session: Session = Depends(db),
):

    interview = session.get(
        Interview,
        interview_id,
    )

    if (
        not interview
        or interview.user_id != user.id
    ):

        raise HTTPException(
            status_code=404,
            detail="Interview not found",
        )

    # ------------------------------------------------------------
    # Build Answer Reviews
    # ------------------------------------------------------------

    reviews = []

    for answer in interview.answers:

        try:

            evaluation = json.loads(
                answer.evaluation_json
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):

            evaluation = {}

        reviews.append(
            {
                "question": answer.question,

                "answer": answer.answer,

                "evaluation": evaluation,

                "question_source": getattr(answer, "question_source", "gemini"),

                "question_topic": getattr(answer, "question_topic", None),
            }
        )

    # ------------------------------------------------------------
    # Dimension Scores
    # ------------------------------------------------------------

    dimensions = [
        "correctness",
        "technical_knowledge",
        "relevance",
        "clarity",
        "communication",
    ]

    dimension_scores = {}

    for dimension in dimensions:

        values = []

        for review in reviews:

            value = review[
                "evaluation"
            ].get(
                dimension
            )

            try:

                values.append(
                    float(value)
                )

            except (
                TypeError,
                ValueError,
            ):

                pass

        if values:

            dimension_scores[
                dimension
            ] = round(
                sum(values)
                / len(values),
                1,
            )

        else:

            dimension_scores[
                dimension
            ] = 0.0

    # ------------------------------------------------------------
    # Overall Score
    # ------------------------------------------------------------

    overall_score = (
        interview.score
    )

    if overall_score is None:

        overall_values = []

        for review in reviews:

            try:

                overall_values.append(
                    float(
                        review[
                            "evaluation"
                        ].get(
                            "overall_score",
                            0,
                        )
                    )
                )

            except (
                TypeError,
                ValueError,
            ):

                pass

        if overall_values:

            overall_score = round(
                (
                    sum(
                        overall_values
                    )
                    / len(
                        overall_values
                    )
                )
                * 10
            )

        else:

            overall_score = 0

    # ------------------------------------------------------------
    # Gemini Final Report
    # ------------------------------------------------------------

    ai_report = (
        llm_service.generate_report(
            role=interview.role,

            experience=(
                interview.experience
            ),

            interview_type=(
                interview.interview_type
            ),

            difficulty=(
                interview.difficulty
            ),

            reviews=reviews,

            overall_score=(
                overall_score
            ),
        )
    )

    # ------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------

    if ai_report is not None:

        ai_report_payload = (
            ai_report.model_dump()
        )

        report_mode = "gemini"

    else:

        strong_dimensions = [
            dimension.replace(
                "_",
                " ",
            ).title()

            for dimension, score
            in dimension_scores.items()

            if score >= 7
        ]

        weak_dimensions = [
            dimension.replace(
                "_",
                " ",
            ).title()

            for dimension, score
            in dimension_scores.items()

            if score <= 4
        ]

        if not strong_dimensions:

            strong_dimensions = [
                "No clearly strong dimension yet"
            ]

        if not weak_dimensions:

            weak_dimensions = [
                "Continue improving consistency"
            ]

        ai_report_payload = {

            "summary": (
                "The interview shows a mixed "
                "level of performance. Continue "
                "practicing the areas with "
                "lower scores."
            ),

            "strongest_skills": (
                strong_dimensions
            ),

            "skill_gaps": (
                weak_dimensions
            ),

            "technical_assessment": (
                "Review the technical concepts "
                "covered during the interview "
                "and practice explaining them "
                "with examples."
            ),

            "communication_assessment": (
                "Use concise, structured answers "
                "with clear technical terminology."
            ),

            "problem_solving_assessment": (
                "Practice explaining the steps "
                "used to solve technical problems."
            ),

            "role_readiness": (
                "Developing. More targeted "
                "practice is recommended before "
                "the next interview."
            ),

            "improvement_plan": [
                "Review weak technical concepts.",
                "Practice concise technical explanations.",
                "Use concrete examples in answers.",
            ],

            "seven_day_plan": [
                "Day 1: Review weak concepts.",
                "Day 2: Solve basic technical problems.",
                "Day 3: Practice explaining solutions.",
                "Day 4: Practice role-specific questions.",
                "Day 5: Complete a mock interview.",
                "Day 6: Review mistakes and missing concepts.",
                "Day 7: Take another full practice interview.",
            ],

            "recommendation": (
                "Focus on the lowest-scoring "
                "areas and practice answering "
                "with a clear concept, "
                "explanation, and example."
            ),
        }

        report_mode = "fallback"

    # ------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------

    if overall_score >= 80:

        readiness = "Strong"

    elif overall_score >= 60:

        readiness = "Good"

    elif overall_score >= 40:

        readiness = "Developing"

    else:

        readiness = "Needs Improvement"

    # ------------------------------------------------------------
    # Evaluation provenance
    # ------------------------------------------------------------
    evaluation_sources = {}
    for review in reviews:
        source = str(review.get("evaluation", {}).get("source", "unknown"))
        evaluation_sources[source] = evaluation_sources.get(source, 0) + 1

    question_sources = {}
    for review in reviews:
        # Prefer persisted provenance; fall back to bank lookup for legacy rows.
        source = str(review.get("question_source") or "").strip()
        if not source:
            try:
                from .interview.question_bank import find_question
                source = "question-bank" if find_question(review.get("question", "")) else "ai-or-planner"
            except Exception:
                source = "unknown"
        question_sources[source] = question_sources.get(source, 0) + 1

    # ------------------------------------------------------------
    # Response
    # ------------------------------------------------------------

    return {

        "interview": interview_out(
            interview
        ),

        "overall_score": (
            overall_score
        ),

        "readiness": readiness,

        "dimension_scores": (
            dimension_scores
        ),

        "strong_areas": (
            ai_report_payload[
                "strongest_skills"
            ]
        ),

        "weak_areas": (
            ai_report_payload[
                "skill_gaps"
            ]
        ),

        "ai_assessment": {

            "summary": (
                ai_report_payload[
                    "summary"
                ]
            ),

            "technical_assessment": (
                ai_report_payload[
                    "technical_assessment"
                ]
            ),

            "communication_assessment": (
                ai_report_payload[
                    "communication_assessment"
                ]
            ),

            "problem_solving_assessment": (
                ai_report_payload[
                    "problem_solving_assessment"
                ]
            ),

            "role_readiness": (
                ai_report_payload[
                    "role_readiness"
                ]
            ),
        },

        "improvement_plan": (
            ai_report_payload[
                "improvement_plan"
            ]
        ),

        "seven_day_plan": (
            ai_report_payload[
                "seven_day_plan"
            ]
        ),

        "recommendation": (
            ai_report_payload[
                "recommendation"
            ]
        ),

        "questions_reviewed": len(
            reviews
        ),

        "reviews": reviews,

        "report_mode": report_mode,
        "evaluation_sources": evaluation_sources,
        "question_sources": question_sources,
        "evaluation_methodology": {
            "gemini": "Structured Gemini evaluation when available.",
            "local": "Deterministic NLP evaluator with question-bank concept coverage and TF-IDF similarity when Gemini is unavailable.",
            "readiness_ml": "Random Forest readiness model consumes the same normalized interview features, so final scoring continues during LLM outages.",
        },
    }

# ================================================================
# AI / ML Assessment
# ================================================================

@app.get("/api/interviews/{interview_id}/ai-assessment")
def ai_assessment(
    interview_id: str,
    user: User = Depends(current_user),
    session: Session = Depends(db),
):
    """Return explainable NLP + ML signals for an interview.

    The LLM remains responsible for rich qualitative feedback. This endpoint
    adds deterministic NLP features and a trained Random Forest readiness
    predictor so the system has an independently evaluated ML component.
    """
    interview = session.get(Interview, interview_id)
    if not interview or interview.user_id != user.id:
        raise HTTPException(status_code=404, detail="Interview not found")

    analyses = []
    evaluations = []
    for answer in interview.answers:
        try:
            evaluation = json.loads(answer.evaluation_json)
        except (json.JSONDecodeError, TypeError):
            evaluation = {}
        evaluations.append(evaluation)
        analyses.append(analyze_answer(answer.question, answer.answer, evaluation))

    aggregate = aggregate_readiness(analyses, evaluations)
    skill_scores = {}
    for key, label in [("technical_score", "Technical"), ("correctness", "Correctness"),
                       ("relevance", "Relevance"), ("clarity", "Clarity"),
                       ("communication", "Communication")]:
        vals = []
        for a in evaluations:
            try: vals.append(float(a.get(key, 0)) * 10)
            except (TypeError, ValueError): pass
        skill_scores[label] = round(sum(vals) / len(vals), 1) if vals else 0

    priorities = sorted(skill_scores.items(), key=lambda x: x[1])[:3]
    recommendations = [f"Practice {name}: current signal {score:.0f}/100." for name, score in priorities if score < 75]
    if not recommendations:
        recommendations.append("Maintain your current preparation and increase question difficulty gradually.")

    return {
        "interview_id": interview_id,
        "model": "Random Forest Regressor",
        "readiness": aggregate,
        "dimension_scores": skill_scores,
        "answer_analytics": analyses,
        "priority_areas": [x[0] for x in priorities],
        "recommendations": recommendations,
        "methodology": {
            "nlp": ["TF-IDF semantic similarity", "question keyword coverage", "filler-word analysis"],
            "ml": "Random Forest readiness prediction trained from data/readiness_dataset.csv",
            "note": "The bundled dataset is a bootstrap/demo dataset. Replace it with consented or public domain data before publishing academic results.",
        },
    }


@app.get("/api/ai/model-metrics")
def ai_model_metrics(user: User = Depends(current_user)):
    """Return reproducible holdout metrics for the readiness model."""
    _, metrics = get_readiness_model()
    return metrics
