# Zyro HR Help Desk (RAG chatbot)

Streamlit chatbot that answers employee HR questions from Zyro Dynamics' policy PDFs.

**Stack:** PyPDFDirectoryLoader · RecursiveCharacterTextSplitter · all-MiniLM-L6-v2 · FAISS (MMR retrieval) · Groq (openai/gpt-oss-120b) · LangChain LCEL · LangSmith tracing

**Features:** source citations on every answer · two-layer out-of-scope guardrail (topic classifier + grounded-answer check) · polite refusals

## Run locally
```
pip install -r requirements.txt
export GROQ_API_KEY=...        # optional: LANGCHAIN_API_KEY=...
streamlit run app.py
```
On Streamlit Cloud, add `GROQ_API_KEY` and `LANGCHAIN_API_KEY` under App settings > Secrets.
