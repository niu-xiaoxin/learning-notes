# 上传笔记到 GitHub
Set-Location D:\learning-notes
$git = "C:\Program Files\Git\cmd\git.exe"

& $git add .
& $git status

Write-Host ""
$msg = Read-Host "请输入本次备注（如：添加微积分笔记）"
& $git commit -m $msg
& $git push

Write-Host ""
Write-Host "上传完成！" -ForegroundColor Green
Read-Host "按回车键关闭"
