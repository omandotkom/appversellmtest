"""Start the dashboard. Usage:  python run.py"""
import uvicorn

from app import config, db

if __name__ == "__main__":
    config.ensure_dirs()
    db.init()
    uvicorn.run(
        "app.main:app",
        host=config.get("HOST", "127.0.0.1"),
        port=config.get_int("PORT", 8000),
        log_level="info",
    )
