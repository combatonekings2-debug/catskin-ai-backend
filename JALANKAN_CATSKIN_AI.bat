@echo off
chcp 65001 >nul
title CatSkin AI - Command Center
color 0B

rem Deteksi root folder proyek (bisa dijalankan dari root atau dari dalam folder frontend)
set "BASE_DIR=%~dp0"
if exist "%~dp0..\backend" (
    set "BASE_DIR=%~dp0..\"
)

:START_SERVERS
cls
echo.
echo  =============================================================================
echo                    CATSKIN AI - COMMAND CENTER LAUNCHER
echo              Sistem Diagnosa Penyakit Kulit Kucing (MobileNetV2)
echo  =============================================================================
echo.
echo                       /\_/\  
echo                      ( o.o )   "Sedang memeriksa dan menyalakan Server AI..."
echo                      (  -  )
echo.
echo  -----------------------------------------------------------------------------

rem 1. Periksa apakah Backend Port 8000 sudah aktif
powershell -NoProfile -Command "(Test-NetConnection -Port 8000 -ComputerName 127.0.0.1 -InformationLevel Quiet)" | findstr /I "True" >nul
if errorlevel 1 (
    echo   [1/3] Menyalakan Backend FastAPI - Python MobileNetV2...
    start "CatSkin AI - Backend Server" /d "%BASE_DIR%backend" cmd /k ".\venv\Scripts\python.exe main.py"
    ping 127.0.0.1 -n 4 >nul
) else (
    echo   [1/3] Backend FastAPI sudah AKTIF di port 8000.
)

rem 2. Periksa apakah Cloudflare Tunnel sudah aktif
tasklist /FI "IMAGENAME eq cloudflared.exe" 2>nul | findstr /I "cloudflared.exe" >nul
if errorlevel 1 (
    echo   [2/3] Menyalakan Cloudflare Tunnel - Protokol HTTP/2...
    start "CatSkin AI - Cloudflare Tunnel" /d "%BASE_DIR%frontend" cmd /k ".\cloudflared.exe tunnel --protocol http2 --url http://127.0.0.1:8000 --metrics 127.0.0.1:20241"
    ping 127.0.0.1 -n 3 >nul
) else (
    echo   [2/3] Cloudflare Tunnel sudah AKTIF di latar belakang.
)

rem 3. Jalankan auto-sync URL Tunnel ke Vercel jika alamat berganti
echo   [3/3] Memeriksa status URL Cloudflare Tunnel dan Sinkronisasi Vercel...
powershell -NoProfile -ExecutionPolicy Bypass -File "%BASE_DIR%sync_tunnel.ps1" -RootDir "%BASE_DIR%"

:DASHBOARD
cls
set "STARTUP_STATUS=NONAKTIF"
if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\CatSkin_AI.lnk" (
    set "STARTUP_STATUS=AKTIF"
)

set "ACTIVE_TUNNEL=Mendeteksi..."
if exist "%BASE_DIR%tunnel_url.txt" (
    set /p ACTIVE_TUNNEL=<"%BASE_DIR%tunnel_url.txt"
)

echo.
echo  =============================================================================
echo                    CATSKIN AI - COMMAND CENTER LAUNCHER
echo              Sistem Diagnosa Penyakit Kulit Kucing (MobileNetV2)
echo  =============================================================================
echo.
echo                       /\_/\  
echo                      ( ='.'= )  "Semua sistem AI telah AKTIF dan siap digunakan!"
echo                      (")_(")
echo.
echo  -----------------------------------------------------------------------------
echo    STATUS SISTEM:
echo    * Backend FastAPI   : ONLINE  - http://127.0.0.1:8000
echo    * Cloudflare Tunnel : ONLINE  - %ACTIVE_TUNNEL%
echo    * Website Frontend  : ONLINE  - https://catskin-ai-frontend.vercel.app
echo    * Auto-Startup PC   : [%STARTUP_STATUS%] - Mulai saat laptop dinyalakan
echo  -----------------------------------------------------------------------------
echo.
echo    PILIHAN MENU:
echo    [1] Buka Website di Browser
echo    [2] Buka Dokumentasi API Swagger (Port 8000)
echo    [3] Cek dan Sinkronkan Ulang URL Tunnel ke Vercel (Auto-Sync)
echo    [4] Restart Backend dan Cloudflare Tunnel
echo    [5] Aktifkan / Matikan Auto-Startup saat Laptop Hidup
echo    [6] Matikan Semua Server dan Tutup Jendela
echo    [7] Tutup Jendela Ini (Server Tetap Berjalan di Background)
echo.
echo  =============================================================================
echo.

choice /c 1234567 /n /m "  >> Tekan angka pilihan [1 - 7]: "

if errorlevel 7 goto KELUAR
if errorlevel 6 goto STOP_ALL
if errorlevel 5 goto TOGGLE_STARTUP
if errorlevel 4 goto RESTART
if errorlevel 3 goto SYNC_TUNNEL
if errorlevel 2 goto OPEN_API
if errorlevel 1 goto OPEN_WEB

:OPEN_WEB
echo.
echo   [OK] Membuka website di browser...
start "" "https://catskin-ai-frontend.vercel.app"
echo.
echo   Website telah dibuka di browser Anda.
echo   Tekan tombol apa saja untuk kembali ke menu...
pause >nul
goto DASHBOARD

:OPEN_API
echo.
echo   [OK] Membuka dokumentasi API di browser...
start "" "http://127.0.0.1:8000/docs"
echo.
echo   Dokumentasi API Swagger telah dibuka.
echo   Tekan tombol apa saja untuk kembali ke menu...
pause >nul
goto DASHBOARD

:SYNC_TUNNEL
echo.
echo   [*] Menjalankan Sinkronisasi URL Tunnel ke Vercel...
powershell -NoProfile -ExecutionPolicy Bypass -File "%BASE_DIR%sync_tunnel.ps1" -RootDir "%BASE_DIR%"
echo.
echo   Tekan tombol apa saja untuk kembali ke menu...
pause >nul
goto DASHBOARD

:RESTART
echo.
echo   [!] Me-restart Backend dan Cloudflare Tunnel...
taskkill /F /IM cloudflared.exe >nul 2>&1
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*main.py*' } | Stop-Process -Force -ErrorAction SilentlyContinue" >nul 2>&1
ping 127.0.0.1 -n 3 >nul
goto START_SERVERS

:TOGGLE_STARTUP
echo.
if exist "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\CatSkin_AI.lnk" goto DISABLE_STARTUP
goto ENABLE_STARTUP

:DISABLE_STARTUP
del /f /q "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\CatSkin_AI.lnk" >nul 2>&1
echo   [OK] Auto-Startup DINONAKTIFKAN! Tidak akan otomatis jalan saat laptop hidup.
echo.
echo   Tekan tombol apa saja untuk kembali ke menu...
pause >nul
goto DASHBOARD

:ENABLE_STARTUP
powershell -NoProfile -Command "$wsh = New-Object -ComObject WScript.Shell; $sc = $wsh.CreateShortcut([System.IO.Path]::Combine($env:APPDATA, 'Microsoft\Windows\Start Menu\Programs\Startup', 'CatSkin_AI.lnk')); $sc.TargetPath = '%BASE_DIR%JALANKAN_CATSKIN_AI.bat'; $sc.WorkingDirectory = '%BASE_DIR%'; $sc.Description = 'CatSkin AI Auto-Startup'; $sc.Save()" >nul 2>&1
echo   [OK] Auto-Startup DIAKTIFKAN! Akan otomatis jalan saat laptop hidup.
echo.
echo   Tekan tombol apa saja untuk kembali ke menu...
pause >nul
goto DASHBOARD

:STOP_ALL
echo.
echo   [!] Menghentikan semua layanan CatSkin AI...
taskkill /F /IM cloudflared.exe >nul 2>&1
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*main.py*' } | Stop-Process -Force -ErrorAction SilentlyContinue" >nul 2>&1
echo   [OK] Semua layanan telah dihentikan. Sampai jumpa!
ping 127.0.0.1 -n 2 >nul
exit

:KELUAR
echo.
echo   [OK] Server tetap aktif di latar belakang. Sampai jumpa!
ping 127.0.0.1 -n 2 >nul
exit