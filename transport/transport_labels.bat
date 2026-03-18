@echo off
setlocal

set "scriptdir=%~dp0"
set "infile=%scriptdir%txt_list.txt"
set "outdir=%scriptdir%labels"

:: Check input file
if not exist "%infile%" (
    echo txt_list.txt not found.
    pause
    exit /b 1
)

:: Create labels folder if it doesn't exist
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

echo Done! Files copied to "labels" folder
pause