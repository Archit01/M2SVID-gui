@echo off
echo Cleaning Python cache files (__pycache__, *.pyc, *.pyo)...

rem Delete all __pycache__ directories recursively
for /d /r . %%d in (__pycache__) do @if exist "%%d" (
    echo Deleting "%%d"
    rd /s /q "%%d"
)

rem Delete all .pyc and .pyo files recursively
del /s /q /f *.pyc *.pyo >nul 2>&1

echo.
echo Cleanup complete!
pause
