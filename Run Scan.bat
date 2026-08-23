@echo off
title Hatch Research Scanner
echo.
echo ========================================
echo   Hatch Research Scanner
echo ========================================
echo.

cd /d "C:\ClaudeProjects\Hatch Research"
python scan.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Scanner failed. Is Python installed?
    pause
    exit /b 1
)

echo.
echo Copying scan_results.json to your Desktop...
copy /Y "scan_results.json" "D:\Users\Megan\OneDrive\Desktop\scan_results.json" >nul
echo Done! scan_results.json is now on your Desktop.
echo.
echo Drag scan_results.json onto the Value Screen tab at:
echo https://meganwood321.github.io/hatch-mobile/
echo.
pause
