Industrial Training FIST Chatbot (Skeleton)

Overview
This is the initial runnable skeleton for your Industrial Training chatbot. It contains:
- Backend: FastAPI app with a health check and a minimal /api/chat endpoint
- Frontend: Static HTML/CSS/JS matching your design (welcome + farewell messages, small bot avatar)
- Config: Single API key via .env, fixed PDF folder path

What works now (Milestone 1)
- Start the server and open the web page
- Send a message; the server returns a placeholder reply (no PDF knowledge yet)
- Multilingual welcome and simple farewell (keyword or idle)

Next milestones (to be implemented)
- OCR + PDF ingestion + embeddings + retrieval (RAG)
- Real answers from your PDFs (no sources shown in UI)

Folder Structure
- server/               # FastAPI backend
- web/                  # Frontend static files
- .env.example          # Copy to .env and fill your single API key

Prerequisites
- Python 3.10+

Setup & Run
1) Create and fill .env
   Copy .env.example to .env and set:
   OPENAI_API_KEY=YOUR_KEY_HERE

2) Install deps
   pip install -r requirements.txt

3) Run backend
   uvicorn server.main:app --reload --host 127.0.0.1 --port 8000

4) Open frontend
   Open web/index.html in your browser (or serve web/ via any static server).

Robot image
Place your robot image at web/assets/bot.png. I will size it small (40px desktop, 32px mobile).

Fixed PDF folder (for later ingestion)
C:\Users\Asus\Desktop\FYP 2\application\需要用到的资料









