@echo off
setlocal

echo =========================================================
echo Setting up Portable Python 3.12 for M2SVID GUI (Windows)
echo Using Python 3.12.9 (Portable) and CUDA 12.8
echo =========================================================

REM Ensure standard Windows utilities are in the PATH (needed for where, curl, tar, etc.)
set PATH=%SystemRoot%\system32;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%

REM Check for Git
git --version >nul 2>&1
if %errorlevel% neq 0 (
    where git >nul 2>&1
    if %errorlevel% neq 0 (
        echo [WARNING] Git not found. Some dependencies may fail to install.
        echo Please install Git from https://git-scm.com/download/win
        pause
    )
)


set PYTHON_DIR=%CD%\python_embed
set PYTHON_EXE=%PYTHON_DIR%\python.exe

if not exist "%PYTHON_DIR%" (
    echo 1. Downloading Portable Python 3.12.9...
    mkdir "%PYTHON_DIR%"
    curl -sSL -o python.zip https://www.python.org/ftp/python/3.12.9/python-3.12.9-embed-amd64.zip
    tar -xf python.zip -C "%PYTHON_DIR%"
    del python.zip

    echo 2. Configuring Portable Python for PIP and project paths...
    powershell -Command "(Get-Content '%PYTHON_DIR%\python312._pth') -replace '#import site', 'import site' | Set-Content '%PYTHON_DIR%\python312._pth'"
    REM Add project root and third_party paths
    echo %CD%>> "%PYTHON_DIR%\python312._pth"
    echo %CD%\third_party\Hi3D-Official>> "%PYTHON_DIR%\python312._pth"
    echo %CD%\third_party\pytorch-msssim>> "%PYTHON_DIR%\python312._pth"

    echo 3. Installing PIP...
    curl -sSL -o get-pip.py https://bootstrap.pypa.io/get-pip.py
    "%PYTHON_EXE%" get-pip.py
    del get-pip.py
) else (
    echo 1. Portable Python 3.12 already exists.
)

echo.
echo 4. Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip

echo.
echo 5. Installing PyTorch ecosystem (CUDA 12.8)...
"%PYTHON_EXE%" -m pip install --no-cache-dir torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    echo Failed to install PyTorch!
    pause
    exit /b %errorlevel%
)

echo.
echo 6. Installing xFormers 0.0.33.post2...
"%PYTHON_EXE%" -m pip install --no-cache-dir xformers==0.0.33.post2 --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    echo    WARNING: xFormers could not be installed.
    echo    The app will use PyTorch native SDPA attention as a fallback.
)

echo.
echo 6.5 Patching xFormers for RTX 50-series (Blackwell) support...
"%PYTHON_EXE%" -c "import os; path = r'python_embed\Lib\site-packages\xformers\ops\fmha\cutlass.py'; content = open(path).read() if os.path.exists(path) else ''; open(path, 'w').write(content.replace('CUDA_MAXIMUM_COMPUTE_CAPABILITY = (9, 0)', 'CUDA_MAXIMUM_COMPUTE_CAPABILITY = (12, 0)')) if content else None; print('✅ Patch Applied (sm_120)') if content else print('⚠️ cutlass.py not found')"

echo.
echo 7. Installing Triton (Windows)...
"%PYTHON_EXE%" -m pip install --no-cache-dir "triton-windows>=3.2.0"
if %errorlevel% neq 0 (
    echo Failed to install triton-windows. continuing anyway...
)

echo.
echo 8. Installing GUI dependencies from requirements_windows.txt...
"%PYTHON_EXE%" -m pip install --no-cache-dir -r requirements_windows.txt
if %errorlevel% neq 0 (
    echo Warning: Some dependencies failed to install. Check logs above.
)

echo.
echo =========================================================
echo Installation complete! 
echo.
echo Note: 
echo - Python is installed locally in the 'python_embed' folder!
echo - No system-wide Python was touched or modified.
echo - PyTorch 2.9.1 (CUDA 12.8) and matching xFormers 0.0.33.post2 are installed.
echo.
echo To run your app in the future, just use run_app.bat which handles everything automatically!
echo =========================================================
endlocal
pause
