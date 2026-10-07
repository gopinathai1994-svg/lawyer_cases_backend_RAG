# Lawyer Corruption RAG Backend - Beginner Flask Version

This backend is intentionally simple.

## What this project does

1. Reads documents from the `documents/` folder.
2. Supports PDF, DOCX and TXT.
3. Splits documents into small chunks.
4. Creates embeddings using the Gemini API.
5. Stores vectors in ChromaDB.
6. Retrieves the most relevant chunks.
7. Sends only retrieved context to Gemini.
8. Returns the RAG answer.
9. Evaluates the answer with:
   - BERTScore
   - ROUGE-1
   - ROUGE-L
   - METEOR
   - Retrieval similarity
10. Reports Gemini token usage, response latency and estimated API cost.

## Project structure

```text
lawyer_corruption_rag_backend/
|
|-- app.py
|-- config.py
|-- gemini_service.py
|-- rag_engine.py
|-- evaluation.py
|-- requirements.txt
|-- .env.example
|-- .gitignore
|-- README.md
|
|-- documents/
|   |-- your_pdf_or_docx_or_txt_files_go_here
|
|-- evaluation/
|
|-- vector_db/
|   |-- Chroma files are created automatically
```

## Step 1 - Create virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

If PowerShell blocks activation, use:

```powershell
.\venv\Scripts\Activate.ps1
```

## Step 2 - Install packages

```bash
pip install -r requirements.txt
```

## Step 3 - Create .env

Copy:

```text
.env.example
```

to:

```text
.env
```

Then put your Gemini API key:

```env
GEMINI_API_KEY=your_real_key_here
```

Do NOT put the API key directly in Python code.

## Step 4 - Put your documents

Put your project documents only inside:

```text
documents/
```

Supported:
- .pdf
- .docx
- .txt

## Step 5 - Start Flask

```bash
python app.py
```

Backend will run at:

```text
http://127.0.0.1:5000
```

## Step 6 - Postman test

### A. Health

GET

```text
http://127.0.0.1:5000/health
```

### B. Check documents

GET

```text
http://127.0.0.1:5000/documents
```

### C. Create vectors

POST

```text
http://127.0.0.1:5000/ingest
```

Body is not required.

### D. Ask question

POST

```text
http://127.0.0.1:5000/chat
```

Body -> raw -> JSON:

```json
{
  "question": "What does the document say about corruption allegations?"
}
```

### E. Evaluate answer

POST

```text
http://127.0.0.1:5000/evaluate
```

Body -> raw -> JSON:

```json
{
  "question": "What does the document say about corruption allegations?",
  "expected_answer": "The document states that the allegation concerns..."
}
```

The response contains:

```text
bert_score_f1
rouge1_f1
rougeL_f1
meteor
retrieval_similarity
overall_score
quality
comment
```

It also contains:

```text
input_tokens
output_tokens
total_tokens
latency_ms
estimated_api_cost_usd
```

## How the score works

This beginner version uses:

```text
BERTScore F1        = 40%
ROUGE-L F1          = 30%
METEOR              = 20%
Retrieval similarity= 10%
```

Overall score:

```text
0.75+  -> GOOD
0.55+  -> MEDIUM
below  -> LOW
```

These thresholds are starting points for your college/project evaluation. They are not universal legal-quality standards.

## Important note about evaluation

BERTScore, ROUGE and METEOR need a reference/expected answer.

Therefore `/evaluate` requires:

```json
{
  "question": "...",
  "expected_answer": "..."
}
```

Without a reference answer, you can still use `/chat`, and the backend will show retrieval similarity, token usage, latency and cost.

## Troubleshooting

### Error: GEMINI_API_KEY is missing

Create `.env` and put the API key.

### Error: Vector database is empty

First call:

```text
POST /ingest
```

### Error: No supported documents found

Put files inside:

```text
documents/
```

### First BERTScore run is slow

BERTScore downloads/loads a transformer model locally on first use. Later runs are usually faster.

### If you change documents

Call:

```text
POST /ingest
```

again.

This version clears the old Chroma collection and rebuilds it from the current `documents/` folder.

## Next step

Once this backend gives good evaluation results, a frontend can call:

```text
POST /chat
```

and display:

```text
answer
sources
```

The evaluation endpoint can later be connected to an admin/evaluation page.
