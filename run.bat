@echo off
echo Set WshShell = CreateObject("WScript.Shell") > %temp%\run_hidden.vbs
echo WshShell.Run "cmd /c cd D:\Biubush\Archives\CodeSpace\D-FINE && D:\Biubush\Archives\CodeSpace\D-FINE\.conda\python.exe D:\Biubush\Archives\CodeSpace\D-FINE\early_stop.py -c D:\Biubush\Archives\CodeSpace\D-FINE\configs\dfine\custom\sar_dfine_s.yml --use-amp --seed=0 --patience=50 --min-epochs=100", 0, false >> %temp%\run_hidden.vbs
wscript %temp%\run_hidden.vbs
