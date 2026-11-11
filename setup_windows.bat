@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 (
  py -3 setup_local.py %*
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo Python 3.12 をインストールしてから実行してください。
    pause
    exit /b 1
  )
  python setup_local.py %*
)
set "TGDTA_SETUP_EXIT=%ERRORLEVEL%"
pause
exit /b %TGDTA_SETUP_EXIT%
