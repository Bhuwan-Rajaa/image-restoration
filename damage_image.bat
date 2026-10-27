@echo off
if "%~1"=="" (
    python "%~dp0damage_image.py"
) else (
    python "%~dp0damage_image.py" "%~1"
)
set "damage_exit=%errorlevel%"
echo.
pause
exit /b %damage_exit%
