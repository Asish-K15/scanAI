from fastapi import FastAPI

from app.routers.predict import router as predict_router


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


app.include_router(predict_router)
