@echo off
chcp 65001 >nul
set GIT="C:\Program Files\Git\cmd\git.exe"
cd /d D:\learning-notes

%GIT% add .
%GIT% status

echo.
set /p msg=请输入本次备注（如：添加微积分笔记）：
%GIT% commit -m "%msg%"
%GIT% push

echo.
echo 上传完成！
pause
