@echo off
title FairGuard AI
cd /d "%~dp0"

REM Generate demo data if missing
if not exist "data\demo_loans.csv" (
    echo Generating demo data...
    python scripts\make_demo_data.py
)

REM Train demo models if missing
if not exist "models\logistic_regression.pkl" (
    echo Training demo models...
    python scripts\train_demo_model.py
)

REM Start FastAPI backend
echo Starting FastAPI backend on http://localhost:8000
start /B "" python api.py

REM Start Streamlit dashboard
echo Starting Streamlit dashboard on http://localhost:8501
start /B "" python -m streamlit run app.py --server.port 8501 --server.headless true

REM Start React frontend if node_modules exists
if exist "frontend\node_modules" (
    echo Starting React frontend on http://localhost:5173
    start /B "" cmd /c "cd frontend && npm run dev"
) else (
    echo.
    echo NOTE: React frontend not started.
    echo Run:  cd frontend && npm install && npm run dev
)

echo.
echo FairGuard AI services are starting...
echo   API:          http://localhost:8000
echo   Dashboard:    http://localhost:8501
echo   React:        http://localhost:5173
echo.
pause
