@echo off
echo === 開始同步到 GitHub ===
git add .
git commit -m "Auto update %date% %time%"

:: 這裡改成 master 以符合你的設定
git push origin master

echo === 更新完成！Streamlit 正在重新部署 ===
pause