@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: 检查参数
if "%~1"=="" (
    echo 用法: run_model.bat [模型规格]
    echo 可用规格: n, s, m, l, x
    echo 示例: run_model.bat l
    exit /b 1
)

:: 设置模型规格
set model_size=%~1
set config_file=sar_dfine_%model_size%.yml

:: 检查配置文件是否存在
if not exist "configs\dfine\custom\%config_file%" (
    echo 错误: 配置文件 configs\dfine\custom\%config_file% 不存在
    exit /b 1
)

:: 创建日志目录（如果不存在）
if not exist "logs" mkdir logs

:: 获取当前日期和时间作为日志文件名的一部分
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\sar_dfine_%model_size%_%datetime:~0,8%_%datetime:~8,6%.log

:: 创建VBS脚本以在后台运行命令
echo Set WshShell = CreateObject("WScript.Shell") > %temp%\run_hidden.vbs
echo WshShell.Run "cmd /c chcp 65001 > nul && cd %~dp0 && .\.conda\python.exe early_stop.py -c configs\dfine\custom\%config_file% --use-amp --seed=0 --patience=50 --min-epochs=100 > %logfile% 2>&1", 0, false >> %temp%\run_hidden.vbs

:: 记录启动信息
echo 启动训练任务，日志将保存到: %logfile%
echo 启动时间: %date% %time% > %logfile%

:: 执行VBS脚本
wscript %temp%\run_hidden.vbs

echo 训练任务已在后台启动，请查看日志文件: %logfile% 