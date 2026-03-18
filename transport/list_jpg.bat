@echo off
set folder=C:\Users\User\Documents\CUAV

dir "%folder%\*.jpg" /s /b > jpg_list.txt

echo Done! Check jpg_list.txt
pause