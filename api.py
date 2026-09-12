from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import run_research


app = FastAPI(
    title="Grounded Research Agent API",
    version="1.0.0",
)


# Allow the local frontend to communicate with the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ResearchRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )


@app.get("/")
def root():
    return {
        "name": "Grounded Research Agent",
        "status": "online",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/research")
def research(request: ResearchRequest):
    try:
        result = run_research(request.question)
        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Research request failed.",
        ) from exc