@echo off
rem One-click test entry (Windows): runs the full unittest suite with the
rem project's own .venv python - no reliance on a global python on PATH.
rem Extra arguments are passed through to run_tests.py, e.g. -v.
rem NOTE: keep this file ASCII-only; cmd.exe parses .bat in the OEM codepage.
rem Also keep the "if" block below free of parentheses and colons - cmd.exe
rem counts them while parsing the block and a stray one breaks the batch.
setlocal
rem Switch the console to UTF-8 so the Chinese output of run_tests.py shows
rem correctly in a double-clicked window - codepage 936 would garble it.
chcp 65001 >nul
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo [run_tests] Not found - "%PY%"
    echo [run_tests] Create the environment first, see the README install section.
    echo     python -m venv .venv
    echo     .venv\Scripts\pip install -r requirements.txt
    exit /b 1
)
"%PY%" "%ROOT%run_tests.py" %*
exit /b %ERRORLEVEL%
