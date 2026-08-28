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

if not exist "%PYTHON_DIR%\Include\Python.h" (
    echo.
    echo 3.5 Downloading Python Development Headers and Libraries for Triton / torch.compile...
    curl -sSL -o py_dev.zip https://www.nuget.org/api/v2/package/python/3.12.9
    if exist "%TEMP%\py_dev_temp" rmdir /S /Q "%TEMP%\py_dev_temp"
    mkdir "%TEMP%\py_dev_temp"
    tar -xf py_dev.zip -C "%TEMP%\py_dev_temp"
    if not exist "%PYTHON_DIR%\Include" mkdir "%PYTHON_DIR%\Include"
    if not exist "%PYTHON_DIR%\libs" mkdir "%PYTHON_DIR%\libs"
    xcopy /E /Y /I "%TEMP%\py_dev_temp\tools\include" "%PYTHON_DIR%\Include" >nul
    xcopy /E /Y /I "%TEMP%\py_dev_temp\tools\libs" "%PYTHON_DIR%\libs" >nul
    copy /Y "%TEMP%\py_dev_temp\tools\libs\python312.lib" "%PYTHON_DIR%\" >nul
    rmdir /S /Q "%TEMP%\py_dev_temp"
    del py_dev.zip
    echo Development headers installed successfully.
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
echo 9. Installing CuPy for GPU-accelerated warping...
REM Install with --no-deps to prevent cupy from pulling in a CPU-only PyTorch
"%PYTHON_EXE%" -m pip install --no-cache-dir --no-deps cupy-cuda12x==13.6.0
if %errorlevel% neq 0 (
    echo    WARNING: CuPy could not be installed.
    echo    GPU-accelerated warping will be unavailable. NumPy fallback will be used.
)

echo.
echo 10. Installing CuPy runtime dependency (fastrlock)...
"%PYTHON_EXE%" -m pip install --no-cache-dir fastrlock==0.8.3
if %errorlevel% neq 0 (
    echo    WARNING: fastrlock install failed. CuPy may not work.
)

echo.
echo 10.5 Patching PyTorch Inductor for Windows Triton and torch.compile compatibility...
"%PYTHON_EXE%" -c "import os, sys; f1=os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', '_inductor', 'runtime', 'static_cuda_launcher.py'); f2=os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', '_inductor', 'runtime', 'triton_heuristics.py'); q=chr(34); s1=q+'i64'+q+': '+q+'l'+q; r1=q+'i64'+q+': '+q+'L'+q+' if os.name == '+q+'nt'+q+' else '+q+'l'+q; s2='(binary.metadata.num_ctas, *binary.metadata.cluster_dims)'; r2='(binary.metadata.num_ctas, *getattr(binary.metadata, '+q+'cluster_dims'+q+', (1, 1, 1)))'; t1=open(f1, 'r').read() if os.path.exists(f1) else ''; open(f1, 'w').write(t1.replace(s1, r1)) if s1 in t1 else None; t2=open(f2, 'r').read() if os.path.exists(f2) else ''; open(f2, 'w').write(t2.replace(s2, r2)) if s2 in t2 else None; print('   PyTorch Inductor Windows patches applied successfully.')"

echo.
echo 11. Re-verifying PyTorch CUDA installation...
REM Re-pin PyTorch in case any dependency pulled a different version
"%PYTHON_EXE%" -m pip install --no-cache-dir torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 2>nul

echo.
echo 12. Running post-install verification...
"%PYTHON_EXE%" -c "import torch; print(f'  PyTorch: {torch.__version__}  CUDA available: {torch.cuda.is_available()}')"
"%PYTHON_EXE%" -c "import numpy; print(f'  NumPy:   {numpy.__version__}')"
REM CuPy JIT test: adds PyTorch lib dir so cupy can find CUDA 12 DLLs (cublas64_12, nvrtc64_120_0 etc.)
"%PYTHON_EXE%" -c "import os, sys; tlib=os.path.join(sys.prefix,'Lib','site-packages','torch','lib'); os.add_dll_directory(tlib) if os.path.isdir(tlib) else None; import cupy as cp; a=cp.array([1.0,2.0,3.0]); print(f'  CuPy:    {cp.__version__}  GPU JIT test: OK (sum={float(cp.sum(a))})')" 2>nul || echo   CuPy:    Not available - warping will use NumPy CPU fallback.

echo.
echo =========================================================
echo Installation complete! 
echo.
echo Note: 
echo - Python is installed locally in the 'python_embed' folder!
echo - No system-wide Python was touched or modified.
echo - PyTorch 2.9.1 (CUDA 12.8) and matching xFormers 0.0.33.post2 are installed.
echo - CuPy 13.6.0 installed for optional GPU-accelerated warping.
echo.
echo To run your app in the future, just use run_app.bat which handles everything automatically!
echo =========================================================
endlocal
pause
