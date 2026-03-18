@echo off
setlocal

set "infile=%~dp0jpg_list.txt"
set "outfile=%~dp0filenames.txt"

if not exist "%infile%" (
    echo jpg_list.txt not found.
    pause
    exit /b 1
)

> "%outfile%" (
    for /f "usebackq delims=" %%L in ("%infile%") do (
        for %%F in ("%%L") do echo(%%~nxF
    )
)

echo Done. Created filenames.txt
pause