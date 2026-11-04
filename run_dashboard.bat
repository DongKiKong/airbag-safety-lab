@echo off
setlocal
cd /d "%~dp0"
echo =================================================
echo    AIRBAG SAFETY LAB - INTEGRATED DASHBOARD
echo =================================================
echo.
where py >nul 2>nul
if %errorlevel%==0 (set "PY=py") else (set "PY=python")
%PY% --version
if errorlevel 1 goto python_error
if not exist "%~dp0app.py" goto file_error
echo.
echo [1/2] Installing or checking dependencies ...
%PY% -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto install_error
echo.
echo [2/2] Launching the Streamlit dashboard ...
%PY% -m streamlit run "%~dp0app.py" --server.address localhost --browser.gatherUsageStats false
if errorlevel 1 goto launch_error
goto end
:python_error
echo Python not found. Install Python 3.10+ and try again.
goto failed
:file_error
echo app.py not found. Extract the full ZIP folder.
goto failed
:install_error
echo Package install failed. Check Python and Internet.
goto failed
:launch_error
echo Streamlit launch failed. Inspect messages above.
goto failed
:failed
pause
:end
endlocal
