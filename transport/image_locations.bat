@echo off
setlocal

set "infile=%~dp0filenames.txt"
set "outfile=%~dp0images_locate.txt"

if not exist "%infile%" (
    echo filenames.txt not found.
    pause
    exit /b 1
)

> "%outfile%" (
    for /f "usebackq delims=" %%A in ("%infile%") do (
        echo images\%%A
    )
)

echo Done. Created images_locate.txt
pause