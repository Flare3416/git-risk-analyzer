# backend/main.py
import uuid
import re
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.tasks import JOBS, analyze_repository_task

app = FastAPI(
    title="Git Risk Analyzer API",
    description="ML-powered bug prediction API for Git repositories",
    version="1.0.0"
)

# CORS configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow Next.js frontend or any local client
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    github_url: str = Field(..., description="The HTTPS URL of the public GitHub repository to analyze")

    @field_validator("github_url")
    @classmethod
    def validate_github_url(cls, v: str) -> str:
        # Simple validation check for github urls
        github_pattern = r"^https:\/\/(www\.)?github\.com\/[a-zA-Z0-9_-]+\/[a-zA-Z0-9_\.-]+\/?$"
        if not re.match(github_pattern, v):
            raise ValueError("Must be a valid public GitHub repository URL (e.g. https://github.com/owner/repo)")
        return v

@app.get("/")
def read_root():
    return {
        "name": "Git Risk Analyzer API",
        "status": "healthy",
        "endpoints": {
            "POST /api/analyze": "Initiate repository analysis",
            "GET /api/jobs/{job_id}": "Poll analysis status and retrieve results"
        }
    }

@app.post("/api/analyze", status_code=202)
def analyze_repository(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    github_url = request.github_url.rstrip("/")
    
    # Initialize the job in our global state dict
    JOBS[job_id] = {
        "job_id": job_id,
        "github_url": github_url,
        "status": "pending",
        "progress": 0,
        "results": None,
        "error": None
    }
    
    # Delegate the analysis task to background execution
    background_tasks.add_task(analyze_repository_task, job_id, github_url)
    
    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]
