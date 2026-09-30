@echo off
start "RAG Backend" cmd /k "cd /d D:\projects\enterprise-rag && call .venv\Scripts\activate.bat && python app.py"
start "RAG Frontend" cmd /k "cd /d D:\projects\enterprise-rag\frontend && npm run dev"
timeout /t 12 /nobreak >nul
start http://localhost:5173
