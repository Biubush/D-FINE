@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: 创建日志目录（如果不存在）
if not exist "logs" mkdir logs

:: 获取当前日期和时间作为日志文件名的一部分
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set timestamp=%datetime:~0,8%_%datetime:~8,6%

:: 定义要运行的模型规格
set models=n s m l x

:: 为每个模型启动训练任务
for %%m in (%models%) do (
    set model_size=%%m
    set config_file=sar_dfine_!model_size!.yml
    set logfile=logs\sar_dfine_!model_size!_%timestamp%.log
    
    :: 检查配置文件是否存在
    if exist "configs\dfine\custom\!config_file!" (
        echo 启动 !model_size! 模型训练，日志将保存到: !logfile!
        
        :: 创建VBS脚本以在后台运行命令
        echo Set WshShell = CreateObject("WScript.Shell") > %temp%\run_hidden_!model_size!.vbs
        echo WshShell.Run "cmd /c chcp 65001 > nul && cd %~dp0 && .\.conda\python.exe early_stop.py -c configs\dfine\custom\!config_file! --use-amp --seed=0 --patience=50 --min-epochs=100 > !logfile! 2>&1", 0, false >> %temp%\run_hidden_!model_size!.vbs
        
        :: 记录启动信息
        echo 启动时间: %date% %time% > !logfile!
        
        :: 执行VBS脚本
        wscript %temp%\run_hidden_!model_size!.vbs
        
        :: 等待一段时间，避免同时启动所有任务
        timeout /t 5 /nobreak > nul
    ) else (
        echo 警告: 配置文件 configs\dfine\custom\!config_file! 不存在，跳过 !model_size! 模型
    )
)

echo 所有训练任务已在后台启动，请查看logs目录下的日志文件 