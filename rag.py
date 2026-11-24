"""Core RAG logic: embeddings (Gemini API), retrieval, and answer generation (Claude API)."""
import json
import os
import urllib.request
from pathlib import Path

import anthropic
import numpy as np

INDEX_DIR = Path(__file__).parent / "index"
EMBED_MODEL = "gemini-embedding-001"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:batchEmbedContents"
EMBED_DIMS = 768
CLAUDE_MODEL = "claude-opus-5-5"
TOP_K = 8

SYSTEM_PROMPT = """你是公司內部的進出口法規與關稅問答助理。只根據使用者訊息中 <sources> 內提供的資料回答。

規則：
- 用繁體中文回答；法規原文、稅則號碼、案號保留英文原樣。
- 每個事實陳述後標註來源編號，例如 [1]、[2][5]。
- 資料中找不到答案時，直接說「提供的資料中找不到」，並說明缺少什麼；不要用你自己的知識補上稅率或判定結果。
- 稅率與關稅措施有時效性。資料出現不同日期或版本時，以最新的更新日期或生效日為準，並寫出該日期；如果新舊資料衝突，要指出衝突。
- 原產地（COO）判定要說明依據的規則或判例，以及判定邏輯（例如實質轉變、最後組裝地）。
- 回答精簡，先給結論，再給依據。"""


def embed(texts, task, titles=None):
    """Embed up to 100 strings with the Gemini API (key in GEMINI_API_KEY). Returns list of vectors.

    task is "RETRIEVAL_DOCUMENT" (index chunks, optionally with titles) or "RETRIEVAL_QUERY" (questions).
    """
    requests = []
    for i, text in enumerate(texts):
        r = {"model": f"models/{EMBED_MODEL}", "content": {"parts": [{"text": text}]},
             "taskType": task, "outputDimensionality": EMBED_DIMS}
        if titles:
            r["title"] = titles[i]
        requests.append(r)
    req = urllib.request.Request(
        EMBED_URL,
        data=json.dumps({"requests": requests}).encode(),
        headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return [e["values"] for e in json.loads(resp.read())["embeddings"]]


class Index:
    def __init__(self):
        self.vectors = np.load(INDEX_DIR / "vectors.npy")
        with open(INDEX_DIR / "chunks.jsonl", encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f]

    def embed_query(self, query):
        q = np.array(embed([query], "RETRIEVAL_QUERY")[0], dtype=np.float32)
        return q / np.linalg.norm(q)

    def search(self, query, k=TOP_K):
        return self.search_vector(self.embed_query(query), k)

    def search_vector(self, q, k=TOP_K):
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


def build_user_message(question, hits):
    sources = "\n\n".join(
        f'<source id="{n}" file="{h["source"]}" location="{h["location"]}">\n{h["text"]}\n</source>'
        for n, h in enumerate(hits, 1)
    )
    return f"<sources>\n{sources}\n</sources>\n\n問題：{question}"


async def stream_answer(client: anthropic.AsyncAnthropic, question, hits):
    """Yield answer text deltas from Claude."""
    async with client.beta.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": build_user_message(question, hits)}],
    ) as stream:
        async for text in stream.text_stream:
            yield text
        final = await stream.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n（模型拒絕回答此問題。）"
    elif final.stop_reason == "max_tokens":
        yield "\n\n（回答超過長度上限，已截斷。）"
