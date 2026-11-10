# IntelliDoc AI

A beginner-friendly PDF question-answering app that demonstrates retrieval-augmented generation (RAG).

## What it does

1. Upload one or more text-based PDFs.
2. Extract text and split it into small, overlapping sections.
3. Use Gemini embeddings to find sections related to your question.
4. Ask Gemini to answer using those sections.
5. Show the source filename, page, and excerpt for each retrieved reference.

The app reads your Gemini API key from the `GEMINI_API_KEY` environment variable and does not save it to a file. Document excerpts and questions are sent to Google Gemini, so do not upload private or confidential documents.

## Run on Windows

You need Python 3.10 or newer and a Gemini API key. Create one in [Google AI Studio](https://aistudio.google.com/apikey).

Open PowerShell in this project folder, set the key for that PowerShell session, then run:

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell prevents activating the virtual environment, you can skip activation and run:

```powershell
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Keep the key and Streamlit command in the same PowerShell window; the environment variable is only available to processes started from that window. The app opens in your browser. Select PDFs and click **Process PDFs** before asking questions. Gemini is used for both embeddings and answers. API access may have usage limits or charges.

AI requests have a 60-second timeout. While asking a question, the app shows whether it is searching the PDF or generating the answer. If a request times out, check your internet connection and Gemini API access before retrying.

## Project map

- `app.py` — the Streamlit interface and Gemini-powered RAG pipeline.
- `requirements.txt` — Python packages the app needs.
- `.gitignore` — excludes virtual environments and local secrets from Git.

## A few RAG terms

- **Chunk:** a short piece of extracted document text.
- **Embedding:** a numeric representation that helps compare meaning.
- **Retrieval:** finding the chunks most relevant to a question.
- **Generation:** having the language model write an answer from those chunks.
- **RAG:** retrieval plus generation; here, retrieved PDF text is provided to the model as evidence.

Scanned/image-only PDFs are not supported yet; they need OCR to turn images into text first.
