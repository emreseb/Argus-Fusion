@echo off
set folder=C:\Users\User\Documents\CUAV\DATASETv3

dir "%folder%\*.txt" /s /b > txt_list.txt

echo Done! Check txt_list.txt
pause