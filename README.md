# 關稅法規問答 RAG MVP

用自然語言（繁體中文）查詢美國稅則、關稅措施與原產地（COO）判定。系統從內部法規 PDF、CBP 判例與 Excel 中檢索相關段落，交給 Claude 撰寫附來源引用的回答。

> 回答僅供參考，不取代正式的關務判斷或報關決策。

## 運作方式

1. **向量化**：問題用本機 Ollama 模型 `embeddinggemma:300m` 轉成 768 維向量
2. **檢索**：與索引中所有段落計算 cosine 相似度，取前 8 段
3. **生成**：Claude（`claude-opus-5-5`）只根據這 8 段回答，每個事實標註來源編號；資料裡找不到就直接說找不到

網頁會即時顯示每個步驟的耗時與檢索到的來源。每次問答都會寫入 `logs.db`（SQLite）。

## 環境需求

- Python 3.14（開發時使用的版本）
- [Ollama](https://ollama.com/)，並下載向量模型：`ollama pull embeddinggemma:300m`
- Anthropic API 金鑰，設定在環境變數 `ANTHROPIC_API_KEY`

## 安裝與啟動（Windows）

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

$env:ANTHROPIC_API_KEY = "<你的金鑰>"   # 或設為系統環境變數
.venv\Scripts\python.exe server.py
```

打開 http://localhost:8000 即可提問。伺服器綁定 `0.0.0.0:8000`，同網段的電腦也能連線。

範例問題：「鋁百葉的鋁葉片從中國進口、在越南組裝，COO 怎麼判定？」

## 更新資料

repo 已附上建好的索引（`index/`），clone 後可以直接啟動。若 `稅則稅率 進出口相關規範/` 裡的文件有新增或修改，重建索引：

```powershell
.venv\Scripts\python.exe ingest.py
```

- 支援 `.pdf` 與 `.xlsx`，其他格式會略過
- `GENERAL RULES OF INTERPRETATION` 那份（878 頁稅則表）刻意排除，內容太雜，會干擾檢索
- 每次都從頭重建，需要 Ollama 正在執行

## 檔案結構

| 路徑 | 說明 |
|---|---|
| `server.py` | FastAPI 伺服器：網頁與 `/api/ask`（NDJSON 串流）|
| `rag.py` | 向量化、檢索、呼叫 Claude 與系統提示詞 |
| `ingest.py` | 讀取文件、切段、建立索引 |
| `static/index.html` | 前端網頁 |
| `index/` | 向量索引（`vectors.npy`、`chunks.jsonl`）|
| `稅則稅率 進出口相關規範/` | 知識庫原始文件 |
| `logs.db` | 問答紀錄（表 `qa_log`）|
| `PRD.md` | 需求與設計文件 |
