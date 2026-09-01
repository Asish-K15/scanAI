from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.routers.predict import router as predict_router


BASE_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = BASE_DIR / "dashboard"


app = FastAPI(
    title="ScanAI API",
    version="0.1.0",
    description="ScanAI animal health screening API.",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "scanai-api",
    }


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(DASHBOARD_DIR / "index.html")


# API routes must be registered BEFORE the root static-file mount.
app.include_router(predict_router)


# Serve frontend assets such as:
# /style.css
# /app.js
# /images/...
app.mount(
    "/",
    StaticFiles(directory=DASHBOARD_DIR, html=True),
    name="dashboard",
)