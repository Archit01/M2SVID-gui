@echo off
echo Configuring Portable Python Environment...
set PYTHON_DIR=%CD%\python_embed
set PYTHON_EXE=%PYTHON_DIR%\python.exe
set PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%SystemRoot%\system32;%SystemRoot%\System32\WindowsPowerShell\v1.0;%PATH%

echo Setting PYTHONPATH...
set PYTHONPATH=.;.\third_party\Hi3D-Official;.\third_party\pytorch-msssim;%PYTHONPATH%

echo Setting Memory Configurations...
set PYTORCH_ALLOC_CONF=max_split_size_mb:128

echo Launching Gradio UI...
"%PYTHON_EXE%" app.py

pause
