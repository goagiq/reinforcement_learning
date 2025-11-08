@echo off
REM Install CUDA-enabled PyTorch for Windows
REM Your system has CUDA 12.2, so we'll install CUDA 12.1 build (compatible)

echo ============================================================
echo Installing CUDA-enabled PyTorch
echo ============================================================
echo.
echo Your system has CUDA 12.2+
echo Installing PyTorch with CUDA 12.4 support (works with CUDA 12.2+)
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

echo Uninstalling CPU-only PyTorch...
pip uninstall -y torch torchvision torchaudio

echo.
echo Installing CUDA-enabled PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo ============================================================
echo Installation complete!
echo ============================================================
echo.
echo Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if hasattr(torch.version, \"cuda\") else \"None\"}')"
echo.
pause

