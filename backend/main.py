"""
main.py — FastAPI application.

Routes:
  POST /api/analyze-sender  → Mode 1: ICP + value prop
  POST /api/analyze-target  → Mode 2: fit eval + emails + claim map
  GET  /api/cache           → inspect cached leads (debug)
  DELETE /api/cache/{domain} → clear a cached entry

Design: thin route layer. All business logic lives in the agents.
The API is the contract; the agents are the implementation.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agents.sender_agent import analyze_sender
from agents.target_agent import analyze_target
from tools.cache import _load, _domain_key, _save

app = FastAPI(title="Artisan Outbound Intelligence", version="1.0.0")

# CORS — allows the frontend (any origin locally) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────

class SenderRequest(BaseModel):
    url: str


class TargetRequest(BaseModel):
    sender_url: str
    target_url: str
    role: str
    seniority: str


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/api/analyze-sender")
async def analyze_sender_route(req: SenderRequest):
    """
    Mode 1: Fetch and analyze a sender company's public pages.
    Returns value proposition and structured ICP definition.
    """
    if not req.url or len(req.url.strip()) < 4:
        raise HTTPException(status_code=400, detail="Valid company URL required.")
    try:
        result = await analyze_sender(req.url.strip())
        return result
    except Exception as e:
        print(f"[main] analyze_sender error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze-target")
async def analyze_target_route(req: TargetRequest):
    """
    Mode 2: Research a target company + persona against sender's ICP.
    Returns fit score, two outbound emails, and a claim map.
    """
    for field, val in [("sender_url", req.sender_url), ("target_url", req.target_url),
                       ("role", req.role), ("seniority", req.seniority)]:
        if not val or not val.strip():
            raise HTTPException(status_code=400, detail=f"{field} is required.")
    try:
        result = await analyze_target(
            sender_url=req.sender_url.strip(),
            target_url=req.target_url.strip(),
            role=req.role.strip(),
            seniority=req.seniority.strip(),
        )
        return result
    except Exception as e:
        print(f"[main] analyze_target error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/cache")
async def get_cache():
    """Inspect all cached leads. Useful for demos and debugging."""
    data = _load()
    return {
        "total_entries": len(data),
        "domains": list(data.keys()),
        "entries": data,
    }


@app.delete("/api/cache/{domain:path}")
async def clear_cache_entry(domain: str):
    """Clear a specific cached entry so it gets re-fetched."""
    data = _load()
    key = _domain_key(domain)
    if key in data:
        del data[key]
        _save(data)
        return {"deleted": key}
    raise HTTPException(status_code=404, detail=f"No cache entry for: {domain}")


# ── Serve frontend ────────────────────────────────────────────────────────────

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")

if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
