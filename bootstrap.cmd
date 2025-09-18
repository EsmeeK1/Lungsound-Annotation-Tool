@echo off
setlocal ENABLEDELAYEDEXPANSION

REM === Config ===
set "VENV_DIR=.venv"   REM of: venv
for %%I in (.) do set "PROJ=%%~nxI"

REM === Python check ===
where python >nul 2>nul || (echo [FOUT] Python niet gevonden & exit /b 1)

REM === Venv aanmaken (indien nodig) ===
if not exist "%VENV_DIR%" (
  echo [INFO] Venv aanmaken in "%CD%\%VENV_DIR%"...
  python -m venv "%VENV_DIR%" || (echo [FOUT] venv aanmaken faalde & exit /b 1)
)

REM === Venv activeren via CMD (geen execution policy gezeik) ===
call "%VENV_DIR%\Scripts\activate.bat" || (echo [FOUT] kon venv niet activeren & exit /b 1)

REM === Forceer publieke PyPI en neutraliseer private indexen ===
set "PIP_INDEX_URL=https://pypi.org/simple"
set "PIP_EXTRA_INDEX_URL="
set "PIP_TRUSTED_HOST="
set "PIP_CONFIG_FILE="

REM (optioneel) laat pip even zien welke index gebruikt wordt
echo [INFO] PIP_INDEX_URL=%PIP_INDEX_URL%

REM === Pip upgraden en requirements installeren vanaf PyPI ===
python -m pip install --upgrade pip
if exist requirements.txt (
  echo [INFO] Requirements installeren vanaf PyPI...
  pip install --no-cache-dir --upgrade --only-binary=:all: -r requirements.txt || (
    echo [FOUT] Requirements installeren faalde & exit /b 1
  )
) else (
  echo [WAARSCHUWING] requirements.txt niet gevonden — overslaan.
)

REM === Jupyter kernel (optioneel, handig voor notebooks) ===
python -m pip install --upgrade ipykernel
python -m ipykernel install --user --name "%PROJ%" --display-name "Python - %PROJ% venv"

REM === VS Code settings.json schrijven ===
if not exist ".vscode" mkdir ".vscode" >nul 2>nul
> ".vscode\settings.json" (
  echo {
  echo   "python.analysis.typeCheckingMode": "basic",
  echo   "python.defaultInterpreterPath": "${workspaceFolder}/%VENV_DIR%/Scripts/python.exe",
  echo   "python.envFile": "${workspaceFolder}/.env",
  echo   "editor.formatOnPaste": false,
  echo   "editor.formatOnSave": true,
  echo   "editor.formatOnType": true,
  echo   "files.trimTrailingWhitespace": true,
  echo   "[python]": {
  echo     "editor.codeActionsOnSave": { "source.organizeImports": "explicit" },
  echo     "editor.formatOnSave": true,
  echo     "editor.defaultFormatter": "ms-python.python"
  echo   },
  echo   "python.languageServer": "Pylance",
  echo   "python.testing.pytestEnabled": true,
  echo   "[sql]": {
  echo     "editor.formatOnSave": false,
  echo     "editor.formatOnType": false,
  echo     "editor.formatOnPaste": false
  echo   },
  echo   "python.analysis.extraPaths": [ "./dashboards/scripts" ],
  echo   "python.terminal.activateEnvironment": true
  echo }
)

echo [KLAAR] Venv + PyPI install + VS Code settings geregeld.
exit /b 0