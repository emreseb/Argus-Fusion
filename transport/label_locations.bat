@echo off
setlocal

set "infile=%~dp0labelnames.txt"
set "outfile=%~dp0labels_locate.txt"

if not exist "%infile%" (
    echo labelnames.txt not found.
    pause
    exit /b 1
)

> "%outfile%" (
    for /f "usebackq delims=" %%A in ("%infile%") do (
        echo labels\%%A
    )
)

echo Done. Created labels_locate.txt
pause