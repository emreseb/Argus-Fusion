@echo off
setlocal

set "scriptdir=%~dp0"
set "infile=%scriptdir%jpg_list.txt"
set "outdir=%scriptdir%images"

:: Check input file
if not exist "%infile%" (
    echo jpg_list.txt not found.
    pause
    exit /b 1
)

:: Create images folder if it doesn't exist
if not exist "%outdir%" (
    mkdir "%outdir%"
)

:: Copy files
for /f "usebackq delims=" %%L in ("%infile%") do (
    if exist "%%L" (
        copy "%%L" "%outdir%\" >nul
    ) else (
        echo File not found: %%L
    )
)

echo Done! Files copied to "images" folder
pause