"""Entry point to launch Mini-Simon FastAPI dashboard.

Run with:
    python run.py
Then open http:/553 /localhost:8000 in your browser.
"""

import uvicorn


def main() -> None:
    uvicorn.run(
        "web_main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
    