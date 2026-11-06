# sync_tunnel.ps1 - Sinkronisasi Otomatis URL Cloudflare Tunnel ke Vercel & GitHub
param(
    [string]$RootDir = $PSScriptRoot
)

$ErrorActionPreference = "SilentlyContinue"

# Deteksi root folder jika skrip dijalankan dari dalam subfolder frontend
if (Test-Path (Join-Path $PSScriptRoot "..\backend")) {
    $RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$frontendDir = if (Test-Path (Join-Path $RootDir "frontend\vercel.json")) { Join-Path $RootDir "frontend" } else { $RootDir }
$vercelJsonPath = Join-Path $frontendDir "vercel.json"
$apiJsPath = Join-Path $frontendDir "src\services\api.js"
$adminApiJsPath = Join-Path $frontendDir "src\admin\api.js"
$cacheFile = Join-Path $RootDir "tunnel_url.txt"

Write-Host "   [*] Mendeteksi alamat URL Cloudflare Tunnel..." -ForegroundColor Cyan

$tunnelUrl = $null
for ($i = 1; $i -le 15; $i++) {
    try {
        $metrics = (curl.exe -s --max-time 2 http://127.0.0.1:20241/metrics) -split "`n"
        foreach ($line in $metrics) {
            if ($line -match 'userHostname="(https://[a-z0-9\-]+\.trycloudflare\.com)"') {
                $tunnelUrl = $matches[1]
                break
            }
        }
        if ($tunnelUrl) { break }
    } catch {}
    Start-Sleep -Seconds 1
}

if (-not $tunnelUrl) {
    Write-Host "   [!] Belum dapat mendeteksi URL Tunnel (Cloudflare masih booting)." -ForegroundColor Yellow
    exit 0
}

Write-Host "   [OK] Cloudflare Tunnel Aktif: $tunnelUrl" -ForegroundColor Green
Set-Content -Path $cacheFile -Value $tunnelUrl -Encoding UTF8

$needsUpdate = $false
if (Test-Path $vercelJsonPath) {
    $currentVercel = Get-Content $vercelJsonPath -Raw
    if ($currentVercel -notmatch [regex]::Escape($tunnelUrl)) {
        $needsUpdate = $true
    }
} else {
    $needsUpdate = $true
}

if (-not $needsUpdate) {
    Write-Host "   [OK] URL Vercel sudah sinkron dengan Cloudflare Tunnel saat ini." -ForegroundColor Green
    exit 0
}

Write-Host "   [!] Terdeteksi perubahan URL Tunnel! Memulai auto-sync ke GitHub & Vercel..." -ForegroundColor Yellow

# Update vercel.json
if (Test-Path $vercelJsonPath) {
    $vContent = Get-Content $vercelJsonPath -Raw
    $vUpdated = $vContent -replace 'https://[a-z0-9\-]+\.trycloudflare\.com', $tunnelUrl
    Set-Content -Path $vercelJsonPath -Value $vUpdated -Encoding UTF8
    Write-Host "   [1/3] vercel.json diperbarui." -ForegroundColor Cyan
}

# Update api.js
if (Test-Path $apiJsPath) {
    $aContent = Get-Content $apiJsPath -Raw
    $aUpdated = $aContent -replace 'https://[a-z0-9\-]+\.trycloudflare\.com', $tunnelUrl
    Set-Content -Path $apiJsPath -Value $aUpdated -Encoding UTF8
    Write-Host "   [2/3] src/services/api.js diperbarui." -ForegroundColor Cyan
}

# Update admin api.js
if (Test-Path $adminApiJsPath) {
    $admContent = Get-Content $adminApiJsPath -Raw
    $admUpdated = $admContent -replace 'https://[a-z0-9\-]+\.trycloudflare\.com', $tunnelUrl
    Set-Content -Path $adminApiJsPath -Value $admUpdated -Encoding UTF8
    Write-Host "   [3/3] src/admin/api.js diperbarui." -ForegroundColor Cyan
}

# Git commit and push
Write-Host "   [*] Mendorong pembaruan ke GitHub (memicu auto-deploy Vercel)..." -ForegroundColor Cyan
git -C "$frontendDir" add vercel.json src/services/api.js src/admin/api.js
git -C "$frontendDir" commit -m "Auto-update Cloudflare Tunnel URL to $tunnelUrl"
git -C "$frontendDir" push origin main

Write-Host "   [SUKSES] Sinkronisasi selesai! Website Vercel otomatis terhubung ke backend baru." -ForegroundColor Green
