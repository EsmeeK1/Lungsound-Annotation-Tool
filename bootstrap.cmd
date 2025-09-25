@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ===== Config =====
set "VENV_DIR=.venv"
for %%I in (.) do set "PROJ=%%~nxI"

REM ===== Vereist: Python Launcher =====
where py >nul 2>nul || (
  echo [FOUT] Python launcher 'py' niet gevonden. Installeer Python met 'py'.
  exit /b 1
)

REM ===== Nieuwste Python 3.x bepalen via py -3 -V (bijv. 'Python 3.13.7') =====
for /f "tokens=2 delims= " %%V in ('py -3 -V') do set "PY3_FULL=%%V"
for /f "tokens=1,2 delims=." %%A in ("%PY3_FULL%") do set "DESIRED_VER=%%A.%%B"
echo [INFO] Nieuwste Python 3.x (py -3): %DESIRED_VER%  (vol: %PY3_FULL%)

REM ===== Huidige venv-versie bepalen (indien aanwezig) =====
if exist "%VENV_DIR%\Scripts\python.exe" (
  for /f "tokens=2 delims= " %%V in ('"%VENV_DIR%\Scripts\python.exe" -V') do set "CUR_FULL=%%V"
  for /f "tokens=1,2 delims=." %%A in ("%CUR_FULL%") do set "CUR_VER=%%A.%%B"
)

REM ===== Indien venv bestaat en oudere minor heeft: verwijderen =====
if defined CUR_VER (
  if not "%CUR_VER%"=="%DESIRED_VER%" (
    echo [INFO] Oude venv Python %CUR_VER% -> %DESIRED_VER%  (verwijderen)
    rmdir /s /q "%VENV_DIR%"
  ) else (
    echo [INFO] Bestaande venv is al Python %CUR_VER%
  )
)

REM ===== Venv aanmaken met nieuwste Python 3 =====
if not exist "%VENV_DIR%" (
  echo [INFO] Venv aanmaken met py -3...
  py -3 -m venv "%VENV_DIR%" || (echo [FOUT] venv aanmaken faalde & exit /b 1)
)

REM ===== Venv activeren =====
call "%VENV_DIR%\Scripts\activate.bat" || (echo [FOUT] kon venv niet activeren & exit /b 1)

REM ===== Actieve interpreter tonen =====
for /f "tokens=2 delims= " %%V in ('python -V') do set "PY_FULL=%%V"
echo [INFO] Actieve interpreter: %PY_FULL%
echo [INFO] Pad: %CD%\%VENV_DIR%\Scripts\python.exe

REM ===== Forceer publieke PyPI =====
set "PIP_INDEX_URL=https://pypi.org/simple"
set "PIP_EXTRA_INDEX_URL="
set "PIP_TRUSTED_HOST="
set "PIP_CONFIG_FILE="
echo [INFO] PIP_INDEX_URL=%PIP_INDEX_URL%

REM ===== Pip upgraden en requirements installeren =====
python -m pip install --upgrade pip
if exist requirements.txt (
  echo [INFO] Requirements installeren...
  python -m pip install --no-cache-dir --upgrade --only-binary=:all: -r requirements.txt || (
    echo [FOUT] Requirements installeren faalde & exit /b 1
  )
) else (
  echo [WAARSCHUWING] requirements.txt niet gevonden — overslaan.
)

REM ===== (Optioneel) Jupyter kernel registreren =====
python -m pip install --upgrade ipykernel
python -m ipykernel install --user --name "%PROJ%" --display-name "Python - %PROJ% venv"

REM ===== VS Code settings.json (zonder blok-redirectie) =====
if not exist ".vscode" mkdir ".vscode" >nul 2>nul
> ".vscode\settings.json" echo {
>> ".vscode\settings.json" echo   "python.analysis.typeCheckingMode": "basic",
>> ".vscode\settings.json" echo   "python.defaultInterpreterPath": "${workspaceFolder}/%VENV_DIR%/Scripts/python.exe",
>> ".vscode\settings.json" echo   "python.envFile": "${workspaceFolder}/.env",
>> ".vscode\settings.json" echo   "editor.formatOnSave": true,
>> ".vscode\settings.json" echo   "editor.formatOnType": true,
>> ".vscode\settings.json" echo   "files.trimTrailingWhitespace": true,
>> ".vscode\settings.json" echo   "[python]": {
>> ".vscode\settings.json" echo     "editor.codeActionsOnSave": { "source.organizeImports": "explicit" },
>> ".vscode\settings.json" echo     "editor.formatOnSave": true,
>> ".vscode\settings.json" echo     "editor.defaultFormatter": "ms-python.python"
>> ".vscode\settings.json" echo   },
>> ".vscode\settings.json" echo   "python.languageServer": "Pylance",
>> ".vscode\settings.json" echo   "python.testing.pytestEnabled": true,
>> ".vscode\settings.json" echo   "python.terminal.activateEnvironment": true
>> ".vscode\settings.json" echo }

echo [KLAAR] Venv met nieuwste Python 3.x aangemaakt en ingericht.
exit /b 0
