from fastapi import FastAPI

app = FastAPI(
    title="RiskLattice API",
    description="AI-powered fraud containment intelligence",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "risklattice-api",
        "version": "0.1.0",
    }