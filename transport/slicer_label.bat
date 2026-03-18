@echo off
setlocal

set "infile=%~dp0txt_list.txt"
set "outfile=%~dp0labelnames.txt"

if not exist "%infile%" (
    echo txt_list.txt not found.
    pause
    exit /b 1
)

> "%outfile%" (
    for /f "usebackq delims=" %%L in ("%infile%") do (
        for %%F in ("%%L") do echo(%%~nxF
    )
)

echo Done. Created labelnames.txt
pause