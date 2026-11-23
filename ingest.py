"""Build the vector index: read docs -> chunk -> embed with local Ollama -> index/.

Usage: .venv\\Scripts\\python.exe ingest.py
Re-run whenever documents change; the index is rebuilt from scratch.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pypdf

from rag import INDEX_DIR, embed

DOCS_DIR = Path(__file__).parent / "稅則稅率 進出口相關規範"
EXCLUDE = ["GENERAL RULES OF INTERPRETATION"]  # 878-page HTSUS schedule: too noisy for MVP

CHUNK_CHARS = 1200
OVERLAP = 200
EXCEL_ROWS_PER_CHUNK = 15


def split_text(text):
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= CHUNK_CHARS:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = start + CHUNK_CHARS
        if end < len(text):
            # prefer to break at a sentence or line boundary
            cut = max(text.rfind(s, start + CHUNK_CHARS // 2, end) for s in ("\n", "。", ". "))
            if cut > 0:
                end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - OVERLAP
    return out


def read_pdf(path):
    for page_no, page in enumerate(pypdf.PdfReader(path).pages, 1):
        for piece in split_text(page.extract_text() or ""):
            yield {"location": f"p.{page_no}", "text": piece}


def cell_str(v):
    if v is None:
        return ""
    if hasattr(v, "strftime"):
        return v.strftime("%Y-%m-%d")
    return re.sub(r"\s+", " ", str(v)).strip()


def read_xlsx(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    for ws in wb.worksheets:
        # merged cells only hold a value in their top-left cell; copy it to the whole range
        for rng in list(ws.merged_cells.ranges):
            value = ws.cell(rng.min_row, rng.min_col).value
            ws.unmerge_cells(str(rng))
            for row in ws.iter_rows(min_row=rng.min_row, max_row=rng.max_row,
                                    min_col=rng.min_col, max_col=rng.max_col):
                for c in row:
                    c.value = value

        rows = []
        for r_idx, row in enumerate(ws.iter_rows(values_only=True), 1):
            cells = [cell_str(v) for v in row]
            # drop consecutive duplicates produced by filling merged cells horizontally
            dedup = [c for i, c in enumerate(cells) if c and (i == 0 or c != cells[i - 1])]
            if dedup:
                rows.append((r_idx, " | ".join(dedup)))
        if not rows:
            continue

        # header context = rows up to and including the first row with >= 3 cells
        header_end = next((i for i, (_, t) in enumerate(rows[:6]) if t.count(" | ") >= 2), 0)
        header = "\n".join(t for _, t in rows[:header_end + 1])
        body = rows[header_end + 1:] or rows
        for i in range(0, len(body), EXCEL_ROWS_PER_CHUNK):
            group = body[i:i + EXCEL_ROWS_PER_CHUNK]
            text = f"[工作表：{ws.title}]\n{header}\n" + "\n".join(t for _, t in group)
            yield {"location": f"工作表「{ws.title}」第{group[0][0]}-{group[-1][0]}列", "text": text}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    chunks = []
    for path in sorted(DOCS_DIR.iterdir()):
        if any(x in path.name for x in EXCLUDE):
            print(f"skip   {path.name}")
            continue
        reader = {".pdf": read_pdf, ".xlsx": read_xlsx}.get(path.suffix.lower())
        if not reader:
            print(f"skip   {path.name} (unsupported type)")
            continue
        n = 0
        for c in reader(path):
            chunks.append({"source": path.name, **c})
            n += 1
        print(f"read   {path.name}: {n} chunks")

    print(f"\nembedding {len(chunks)} chunks with Ollama ...")
    vectors = []
    batch = 32
    for i in range(0, len(chunks), batch):
        texts = [f"title: {c['source']} | text: {c['text']}" for c in chunks[i:i + batch]]
        vectors.extend(embed(texts))
        print(f"  {min(i + batch, len(chunks))}/{len(chunks)}", end="\r", flush=True)

    vecs = np.array(vectors, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vecs)
    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nsaved {len(chunks)} chunks, vectors {vecs.shape} -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
