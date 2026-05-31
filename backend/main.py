"""
main.py — FastAPI application.
Simple POST endpoints that return JSON when the agent finishes.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agents.sender_agent import analyze_sender
from agents.target_agent import analyze_target

app = FastAPI(title="Outbound Intelligence Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SenderRequest(BaseModel):
    url: str


class TargetRequest(BaseModel):
    sender_url: str
    sender_icp: dict
    value_prop: str
    target_url: str
    role:       str
    seniority:  str


@app.post("/api/analyze-sender")
async def analyze_sender_route(req: SenderRequest):
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="URL is required")
    try:
        return await analyze_sender(req.url.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-target")
async def analyze_target_route(req: TargetRequest):
    for field, val in [
        ("sender_url", req.sender_url), ("target_url", req.target_url),
        ("role", req.role), ("seniority", req.seniority),
    ]:
        if not val.strip():
            raise HTTPException(status_code=400, detail=f"{field} is required")
    try:
        return await analyze_target(
            sender_url=req.sender_url.strip(),
            sender_icp=req.sender_icp,
            value_prop=req.value_prop.strip(),
            target_url=req.target_url.strip(),
            role=req.role.strip(),
            seniority=req.seniority.strip(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
