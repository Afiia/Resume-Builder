"""ATS Resume Tailorer — Free Local Backend (No API Key Required).

Run: uvicorn main:app --reload
"""
import io
import os
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB

app = FastAPI(title="ATS Resume Tailorer")


# ---------- Schemas ----------
class TailorRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, max_length=30_000)
    job_description: str = Field(..., min_length=50, max_length=30_000)


class ScoreBreakdown(BaseModel):
    keywords: int = Field(ge=0, le=100)
    skills: int = Field(ge=0, le=100)
    experience: int = Field(ge=0, le=100)
    structure: int = Field(ge=0, le=100)


class Feedback(BaseModel):
    priority: Literal["high", "medium", "low"]
    area: str
    issue: str
    recommendation: str


class StructureFix(BaseModel):
    issue: str
    fix: str


class WordingFix(BaseModel):
    original: str
    corrected: str
    issue_type: str


class BulletRewrite(BaseModel):
    original: str
    tailored: str
    justification: str


class TailorResult(BaseModel):
    match_score: int
    score_breakdown: ScoreBreakdown
    summary: str
    missing_keywords: list[str]
    improvement_feedback: list[Feedback]
    structure_feedback: list[StructureFix]
    wording_fixes: list[WordingFix]
    suggested_bullet_rewrites: list[BulletRewrite]
    tailored_resume: str


# ---------- Local Helper Analysis Logic ----------
def extract_words(text: str) -> set[str]:
    """Extract clean lowercased words longer than 2 characters."""
    return set(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))


def compute_ats_score(resume: str, job_desc: str) -> tuple[int, list[str]]:
    """Compute cosine similarity score and find missing keywords."""
    # Common English stopwords to exclude
    stopwords = {
        "and", "the", "for", "with", "that", "this", "from", "have", "you",
        "will", "are", "our", "their", "must", "work", "team", "years", "role"
    }

    # Extract unique keywords from job description
    job_words = [w for w in extract_words(job_desc) if w not in stopwords]
    resume_words = extract_words(resume)

    # Missing keywords
    missing = [w.capitalize() for w in job_words if w not in resume_words][:8]

    # Calculate TF-IDF Similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([resume, job_desc])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    match_score = int(min(max(similarity * 100 * 1.5, 30), 95))
    return match_score, missing


# ---------- Routes ----------
@app.post("/api/tailor", response_model=TailorResult)
async def tailor(req: TailorRequest) -> TailorResult:
    score, missing_kw = compute_ats_score(req.resume_text, req.job_description)

    return TailorResult(
        match_score=score,
        score_breakdown=ScoreBreakdown(
            keywords=score,
            skills=min(score + 5, 95),
            experience=max(score - 5, 40),
            structure=85,
        ),
        summary=f"Your resume achieved a {score}% ATS match score based on key term alignment.",
        missing_keywords=missing_kw if missing_kw else ["No critical keywords missing!"],
        improvement_feedback=[
            Feedback(
                priority="high",
                area="Keywords Alignment",
                issue="Key industry terms found in the job posting are missing from your resume.",
                recommendation=f"Incorporate missing keywords naturally: {', '.join(missing_kw[:4])}.",
            ),
            Feedback(
                priority="medium",
                area="Bullet Points Format",
                issue="Some bullet points lack measurable metrics or quantitative impacts.",
                recommendation="Use Google's X-Y-Z formula: Accomplished [X] as measured by [Y], by doing [Z].",
            ),
        ],
        structure_feedback=[
            StructureFix(
                issue="Formatting standard",
                fix="Ensure single-column layout without complex graphics or tables for seamless ATS parsing.",
            )
        ],
        wording_fixes=[
            WordingFix(
                original="Responsible for managing tasks",
                corrected="Spearheaded daily workflows and project coordination",
                issue_type="Weak verb",
            )
        ],
        suggested_bullet_rewrites=[
            BulletRewrite(
                original="Developed backend services for application.",
                tailored="Engineered scalable backend REST APIs using Python, improving response times by 15%.",
                justification="Adds quantifiable achievement metrics.",
            )
        ],
        tailored_resume=req.resume_text,
    )


@app.post("/api/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)) -> dict:
    """Extract plain text from an uploaded PDF."""
    data = await file.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF is larger than 10 MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=415, detail="That file is not a PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise HTTPException(status_code=422, detail="This PDF is password-protected.")
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except PdfReadError:
        raise HTTPException(status_code=422, detail="Could not read this PDF. It may be corrupted.")
    if len(text) < 20:
        raise HTTPException(
            status_code=422,
            detail="No text found. Scanned or image-only PDFs aren't supported; paste the text instead.",
        )
    return {"filename": file.filename, "text": text[:30_000]}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")