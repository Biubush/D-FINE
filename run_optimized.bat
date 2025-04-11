@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: 创建日志目录（如果不存在）
if not exist "logs" mkdir logs

:: 获取当前日期和时间作为日志文件名的一部分
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set logfile=logs\sar_dfine_optimized_%datetime:~0,8%_%datetime:~8,6%.log

:: 显示训练参数
echo ===== SAR图像目标检测优化训练 =====
echo 配置文件: configs/dfine/custom/sar_dfine_optimized.yml
echo 使用AMP: 是
echo 随机种子: 42
echo 早停耐心: 30
echo 最小训练轮次: 50
echo 最小增益阈值: 0.0005
echo 监视指标: AP50:95
echo 日志文件: %logfile%
echo =====================================

:: 确认是否继续
set /p confirm=是否开始训练 (Y/N)？
if /i "%confirm%" neq "Y" (
    echo 训练已取消
    exit /b
)

:: 创建VBS脚本以在后台运行命令
echo Set WshShell = CreateObject("WScript.Shell") > %temp%\run_optimized.vbs
echo WshShell.Run "cmd /c chcp 65001 > nul && cd D:\Biubush\Archives\CodeSpace\D-FINE && D:\Biubush\Archives\CodeSpace\D-FINE\.conda\python.exe D:\Biubush\Archives\CodeSpace\D-FINE\early_stop.py -c D:\Biubush\Archives\CodeSpace\D-FINE\configs\dfine\custom\sar_dfine_optimized.yml --use-amp --seed=42 --patience=30 --min-delta=0.0005 --min-epochs=50 --monitor=AP50:95 > %logfile% 2>&1", 0, false >> %temp%\run_optimized.vbs

:: 记录启动信息
echo 启动训练任务，日志将保存到: %logfile%
echo 启动时间: %date% %time% > %logfile%
echo ===== SAR图像目标检测优化训练 ===== >> %logfile%
echo 配置文件: configs/dfine/custom/sar_dfine_optimized.yml >> %logfile%
echo 使用AMP: 是 >> %logfile%
echo 随机种子: 42 >> %logfile%
echo 早停耐心: 30 >> %logfile%
echo 最小训练轮次: 50 >> %logfile%
echo 最小增益阈值: 0.0005 >> %logfile%
echo 监视指标: AP50:95 >> %logfile%
echo ===================================== >> %logfile%

:: 执行VBS脚本
wscript %temp%\run_optimized.vbs

echo 训练任务已在后台启动，请查看日志文件: %logfile%
echo 可以使用以下命令实时查看训练进度:
echo powershell -command "Get-Content -Path '%logfile%' -Wait" 