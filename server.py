"""Web server: serves the chat page and a streaming /api/ask endpoint.

Usage: .venv\\Scripts\\python.exe server.py   (listens on 0.0.0.0:8000 for LAN access)
Every question is logged to logs.db (SQLite, table qa_log).
"""
import asyncio
import json
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

LOG_DB = Path(__file__).parent / "logs.db"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
index = Index()
client = anthropic.AsyncAnthropic()


def init_log_db():
    with sqlite3.connect(LOG_DB) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS qa_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    with sqlite3.connect(LOG_DB) as db:
        cols = ", ".join(row)
        db.execute(f"INSERT INTO qa_log ({cols}) VALUES ({', '.join('?' * len(row))})", list(row.values()))


init_log_db()


class Ask(BaseModel):
    question: str


def ms_since(t):
    return round((time.perf_counter() - t) * 1000)


def line(obj):
    return json.dumps(obj, ensure_ascii=False) + "\n"


@app.get("/")
def home():
    return FileResponse("static/index.html")


@app.post("/api/ask")
async def ask(req: Ask):
    async def events():
        # NDJSON stream: "step" events for each RAG stage, one "sources" event,
        # "text" deltas, then "done" (or "error" before "done")
        log = {"created_at": datetime.now().isoformat(timespec="seconds"), "question": req.question}
        answer = []
        t_total = time.perf_counter()
        try:
            # step 1: embed the question with the local Ollama model
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
        except OSError:
            log["error"] = "無法連線到本機 Ollama 向量模型，請確認 Ollama 已啟動。"
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
