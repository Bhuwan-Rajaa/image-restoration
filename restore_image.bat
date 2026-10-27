@echo off
if "%~1"=="" (
    python "%~dp0restore_image.py"
) else (
    python "%~dp0restore_image.py" "%~1"
)
set "restore_exit=%errorlevel%"
echo.
pause
exit /b %restore_exit%
