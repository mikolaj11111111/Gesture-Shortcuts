@echo off
echo ============================================
echo   Building Gesture Shortcuts .exe
echo ============================================
echo.

REM Check if PyInstaller is installed
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [1/4] Installing PyInstaller...
    pip install pyinstaller
) else (
    echo [1/4] PyInstaller already installed
)

echo.
echo [2/4] Killing any running GestureShortcuts.exe...
taskkill /F /IM GestureShortcuts.exe >nul 2>&1
if not errorlevel 1 (
    echo      Killed running instance. Waiting for file locks to release...
    timeout /t 2 /nobreak >nul
) else (
    echo      No running instance found.
)

echo.
echo [3/4] Building executable...
echo      This may take a few minutes...
echo.

pyinstaller gesture_shortcuts.spec --noconfirm

if errorlevel 1 (
    echo.
    echo ============================================
    echo   BUILD FAILED!
    echo ============================================
    pause
    exit /b 1
)

echo.
echo ============================================
echo   BUILD SUCCESSFUL!
echo ============================================
echo.
echo   Executable location:
echo   dist\GestureShortcuts\GestureShortcuts.exe
echo.
echo [4/4] Opening output folder...
explorer dist\GestureShortcuts

pause
