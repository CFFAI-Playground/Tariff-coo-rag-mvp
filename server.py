"""Web server: serves the chat page and a streaming /api/ask endpoint.

Usage: .venv\\Scripts\\python.exe server.py   (listens on 0.0.0.0:8000 for LAN access)
Every question is logged to table qa_log: Postgres when DATABASE_URL is set (Vercel + Neon),
otherwise the local SQLite file logs.db.
"""
import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import anthropic
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import CLAUDE_MODEL, EMBED_MODEL, Index, stream_answer

BASE_DIR = Path(__file__).parent
LOG_DB = BASE_DIR / "logs.db"
DATABASE_URL = os.environ.get("DATABASE_URL")

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
index = Index()
client = anthropic.AsyncAnthropic()


def log_db():
    if DATABASE_URL:
        import psycopg
        return psycopg.connect(DATABASE_URL)
    return sqlite3.connect(LOG_DB)


def init_log_db():
    id_col = "SERIAL PRIMARY KEY" if DATABASE_URL else "INTEGER PRIMARY KEY AUTOINCREMENT"
    with log_db() as db:
        db.execute(f"""CREATE TABLE IF NOT EXISTS qa_log (
            id {id_col},
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            sources_json TEXT,
            answer TEXT,
            error TEXT,
            embed_ms INTEGER,
            retrieve_ms INTEGER,
            first_token_ms INTEGER,
            generate_ms INTEGER,
            total_ms INTEGER)""")


def save_log(row):
    mark = "%s" if DATABASE_URL else "?"
    with log_db() as db:
        cols = ", ".join(row)
        db.execute(f"INSERT INTO qa_log ({cols}) VALUES ({', '.join([mark] * len(row))})", list(row.values()))


init_log_db()


class Ask(BaseModel):
    question: str


def ms_since(t):
    return round((time.perf_counter() - t) * 1000)


def line(obj):
    return json.dumps(obj, ensure_ascii=False) + "\n"


@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/api/ask")
async def ask(req: Ask):
    async def events():
        # NDJSON stream: "step" events for each RAG stage, one "sources" event,
        # "text" deltas, then "done" (or "error" before "done")
        log = {"created_at": datetime.now().isoformat(timespec="seconds"), "question": req.question}
        answer = []
        t_total = time.perf_counter()
        try:
            # step 1: embed the question with the Gemini embedding API
            t = time.perf_counter()
            q = await asyncio.to_thread(index.embed_query, req.question)
            log["embed_ms"] = ms_since(t)
            yield line({"type": "step", "step": "embed", "ms": log["embed_ms"], "model": EMBED_MODEL,
                        "dims": len(q), "preview": [round(float(x), 3) for x in q[:5]]})

            # step 2: cosine similarity against every chunk, keep the top k
            t = time.perf_counter()
            hits = index.search_vector(q)
            log["retrieve_ms"] = ms_since(t)
            yield line({"type": "step", "step": "retrieve", "ms": log["retrieve_ms"],
                        "total": len(index.chunks), "k": len(hits), "top_score": round(hits[0]["score"], 3)})

            sources = [{"id": n, "source": h["source"], "location": h["location"],
                        "text": h["text"], "score": round(h["score"], 3)} for n, h in enumerate(hits, 1)]
            log["sources_json"] = json.dumps(sources, ensure_ascii=False)
            yield line({"type": "sources", "sources": sources})

            # step 3: Claude writes the answer from the retrieved chunks
            t = time.perf_counter()
            async for text in stream_answer(client, req.question, hits):
                if not answer:
                    log["first_token_ms"] = ms_since(t)
                    yield line({"type": "step", "step": "first_token", "ms": log["first_token_ms"], "model": CLAUDE_MODEL})
                answer.append(text)
                yield line({"type": "text", "text": text})
            log["generate_ms"] = ms_since(t)
            yield line({"type": "step", "step": "generate", "ms": log["generate_ms"]})
        except anthropic.AuthenticationError:
            log["error"] = "Claude API 金鑰無效或未設定（ANTHROPIC_API_KEY）。"
        except anthropic.RateLimitError:
            log["error"] = "Claude API 請求過於頻繁，請稍後再試。"
        except anthropic.APIConnectionError:
            log["error"] = "無法連線到 Claude API，請檢查網路。"
        except anthropic.APIStatusError as e:
            log["error"] = f"Claude API 錯誤 {e.status_code}"
        except KeyError as e:
            if e.args != ("GEMINI_API_KEY",):
                raise
            log["error"] = "Gemini API 金鑰未設定（GEMINI_API_KEY）。"
        except OSError:
            log["error"] = "無法呼叫 Gemini 向量模型 API，請檢查金鑰與網路。"
        finally:
            # runs on success, error, and client disconnect, so every question is logged
            log["answer"] = "".join(answer)
            log["total_ms"] = ms_since(t_total)
            save_log(log)
        if log.get("error"):
            yield line({"type": "error", "message": log["error"]})
        yield line({"type": "done", "total_ms": log["total_ms"]})

    return StreamingResponse(events(), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
