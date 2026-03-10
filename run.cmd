@echo off
echo ============================================================
echo  AI Product Workflow - Launcher
echo ============================================================
echo.
echo  1. Run Full Pipeline (Flow)
echo  2. Run Analyst Crew only
echo  3. Run Scientist Crew only
echo  4. Launch Streamlit App
echo  5. Exit
echo.
set /p choice="Select an option (1-5): "

if "%choice%"=="1" goto run_flow
if "%choice%"=="2" goto run_analyst
if "%choice%"=="3" goto run_scientist
if "%choice%"=="4" goto run_app
if "%choice%"=="5" goto end

echo Invalid choice. Exiting.
goto end

:run_flow
echo.
echo Starting Full Pipeline...
call venv\Scripts\activate.bat
python -m flow.main_flow
goto done

:run_analyst
echo.
echo Starting Analyst Crew...
call venv\Scripts\activate.bat
python -m crew_analyst.crew
goto done

:run_scientist
echo.
echo Starting Scientist Crew...
call venv\Scripts\activate.bat
python -m crew_scientist.crew
goto done

:run_app
echo.
echo Launching Streamlit App at http://localhost:8501 ...
call venv\Scripts\activate.bat
streamlit run app\app.py
goto done

:done
echo.
echo Done.
pause

:end
