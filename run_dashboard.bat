@echo off
REM -------------------------------------------
REM FAANG-Level Local Launch for Streamlit Dashboard
REM -------------------------------------------

REM 1. Navigate to this script's folder
cd /d %~dp0

REM 2. Activate virtual environment
call venv\Scripts\activate

REM 3. Run Streamlit dashboard
streamlit run dashboards\streamlit_app.py

REM 4. Keep terminal open after exit
pause