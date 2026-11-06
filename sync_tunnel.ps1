# sync_tunnel.ps1 - Sinkronisasi Otomatis URL Cloudflare Tunnel ke Vercel & GitHub
param(
    [string]$RootDir = ""
)

$ErrorActionPreference = "SilentlyContinue"

if ([string]::IsNullOrWhiteSpace($RootDir)) {
    $RootDir = $PSScriptRoot
}
$RootDir = $RootDir.Trim('"', "'").TrimEnd('\', '/')

# Deteksi root folder jika skrip dijalankan dari dalam subfolder frontend
if (Test-Path (Join-Path $RootDir "..\backend")) {
    $RootDir = (Resolve-Path (Join-Path $RootDir "..")).Path
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

# UTF-8 WITHOUT BOM Encoding (CRUCIAL for Vercel JSON parser)
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

[System.IO.File]::WriteAllText($cacheFile, $tunnelUrl, $utf8NoBom)

$needsUpdate = $false
if (Test-Path $vercelJsonPath) {
    $currentVercel = [System.IO.File]::ReadAllText($vercelJsonPath, [System.Text.Encoding]::UTF8)
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

# Update vercel.json (Without BOM)
if (Test-Path $vercelJsonPath) {
    $vContent = [System.IO.File]::ReadAllText($vercelJsonPath, [System.Text.Encoding]::UTF8).TrimStart([char]0xFEFF)
    $vUpdated = $vContent -replace 'https://[a-z0-9\-]+\.trycloudflare\.com', $tunnelUrl
    [System.IO.File]::WriteAllText($vercelJsonPath, $vUpdated, $utf8NoBom)
    Write-Host "   [1/3] vercel.json diperbarui (UTF-8 No BOM)." -ForegroundColor Cyan
}

# Update api.js (Without BOM)
if (Test-Path $apiJsPath) {
    $aContent = [System.IO.File]::ReadAllText($apiJsPath, [System.Text.Encoding]::UTF8).TrimStart([char]0xFEFF)
    $aUpdated = $aContent -replace 'https://[a-z0-9\-]+\.trycloudflare\.com', $tunnelUrl
    [System.IO.File]::WriteAllText($apiJsPath, $aUpdated, $utf8NoBom)
    Write-Host "   [2/3] src/services/api.js diperbarui." -ForegroundColor Cyan
}

# Update admin api.js (Without BOM)
if (Test-Path $adminApiJsPath) {
    $admContent = [System.IO.File]::ReadAllText($adminApiJsPath, [System.Text.Encoding]::UTF8).TrimStart([char]0xFEFF)
    $admUpdated = $admContent -replace 'https://[a-z0-9\-]+\.trycloudflare\.com', $tunnelUrl
    [System.IO.File]::WriteAllText($adminApiJsPath, $admUpdated, $utf8NoBom)
    Write-Host "   [3/3] src/admin/api.js diperbarui." -ForegroundColor Cyan
}

# Git commit and push
Write-Host "   [*] Mendorong pembaruan ke GitHub (memicu auto-deploy Vercel)..." -ForegroundColor Cyan
git -C "$frontendDir" add vercel.json src/services/api.js src/admin/api.js
git -C "$frontendDir" commit -m "Auto-update Cloudflare Tunnel URL to $tunnelUrl (clean UTF-8)"
git -C "$frontendDir" push origin main

Write-Host "   [SUKSES] Sinkronisasi selesai! Website Vercel otomatis terhubung ke backend baru." -ForegroundColor Green
