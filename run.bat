@echo off
echo 🌱 Ultra Simple YOLO Seed Detection Setup
echo =========================================

echo 🚀 Installing dependencies and running...

python3 run.py
if %errorlevel% neq 0 (
    echo Trying 'python' instead of 'python3'...
    python run.py
)

echo ✅ Done!
pause
