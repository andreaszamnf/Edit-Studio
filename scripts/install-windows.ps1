# Edit Studio - Windows installer (run it with install.bat)
# Installs uv, Python 3.12, PyTorch for your graphics card and the app, then creates shortcuts.

$ErrorActionPreference = "Stop"
$env:UV_LINK_MODE = "copy"  # cache and app on different drives: copying is expected, not a problem
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($el, $en) {
    Write-Host ""
    Write-Host "==> $el" -ForegroundColor Cyan
    Write-Host "    $en" -ForegroundColor DarkGray
}
function Fail($el, $en) {
    Write-Host ""
    Write-Host "X  $el" -ForegroundColor Red
    Write-Host "   $en" -ForegroundColor DarkGray
    exit 1
}
function Run($exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { Fail "Η εντολή απέτυχε: $exe $($argv -join ' ')" "Command failed (exit $LASTEXITCODE)" }
}

Write-Host "Edit Studio - Εγκατάσταση / Setup" -ForegroundColor Magenta

# ---------------------------------------------------------------- the app must be closed (its files are in use)
$ProgressPreference = "SilentlyContinue"
foreach ($port in 8765..8774) {
    $running = $false
    try {
        $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 1 -Uri "http://127.0.0.1:$port/api/system"
        $running = $r.Content -match '"version"'
    } catch { }
    if ($running) { Fail "Το Edit Studio είναι ανοιχτό. Κλείσ' το και ξανατρέξε αυτό το αρχείο." "Edit Studio is running. Close it and run this again." }
}
# A window-less copy may still be running (older versions waited for a download to finish before quitting):
# its files are in use, so stop it.
$venvDir = Join-Path $Root ".venv"
$stale = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
    $_.Path -and $_.Path.StartsWith($venvDir, [System.StringComparison]::OrdinalIgnoreCase) })
if ($stale.Count -gt 0) {
    Write-Host ""
    Write-Host "    Κλείνω το Edit Studio που έτρεχε ακόμα στο παρασκήνιο..." -ForegroundColor Yellow
    Write-Host "    Closing Edit Studio, still running in the background..." -ForegroundColor DarkGray
    $stale | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

# ---------------------------------------------------------------- graphics card
Step "Έλεγχος κάρτας γραφικών..." "Checking the graphics card..."
$gpuInfo = $null
try { $gpuInfo = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null } catch { }
if ($gpuInfo) {
    $gpuInfo | ForEach-Object { Write-Host "    $_" }
    $smi = (& nvidia-smi 2>$null) -join "`n"
    if ($smi -match "CUDA Version:\s*([0-9]+)\.([0-9]+)") {
        $cudaMajor = [int]$Matches[1]; $cudaMinor = [int]$Matches[2]
        Write-Host "    CUDA (driver): $cudaMajor.$cudaMinor"
        $isBlackwell = ($gpuInfo -join " ") -match "RTX 50|RTX PRO|B200|Blackwell"
        if ($isBlackwell -and ($cudaMajor -lt 12 -or ($cudaMajor -eq 12 -and $cudaMinor -lt 8))) {
            Write-Host ""
            Write-Host "!  Ο οδηγός (driver) της NVIDIA είναι παλιός για κάρτα RTX 50xx." -ForegroundColor Yellow
            Write-Host "   Κατέβασε τον τελευταίο από https://www.nvidia.com/Download/index.aspx και ξανατρέξε το install.bat" -ForegroundColor Yellow
            Write-Host "   Your NVIDIA driver is too old for an RTX 50xx card. Update it and run install.bat again." -ForegroundColor DarkGray
        }
    }
} else {
    Write-Host "    Δεν βρέθηκε κάρτα NVIDIA - η εφαρμογή θα τρέχει πολύ αργά στον επεξεργαστή." -ForegroundColor Yellow
    Write-Host "    No NVIDIA GPU found - the app will run (very slowly) on the CPU." -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- uv
Step "Εγκατάσταση του uv (εργαλείο Python)..." "Installing uv (Python tool manager)..."
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) {
    $env:UV_INSTALL_DIR = "$env:USERPROFILE\.local\bin"
    powershell -NoProfile -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    $uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
    if (-not $uv) { Fail "Δεν εγκαταστάθηκε το uv. Έλεγξε τη σύνδεση στο internet." "uv could not be installed. Check your internet connection." }
}
Write-Host "    $uv"

# ---------------------------------------------------------------- python + torch
$py = Join-Path $Root ".venv\Scripts\python.exe"
if (Test-Path $py) {
    Step "Το περιβάλλον Python υπάρχει ήδη." "The Python environment already exists."
} else {
    Step "Δημιουργία περιβάλλοντος Python 3.12..." "Creating the Python 3.12 environment..."
    Run $uv @("venv", ".venv", "--python", "3.12", "--allow-existing")
}

# An update keeps a PyTorch that already works (it is ~3 GB); a missing or broken one is installed again.
$needCuda = if ($gpuInfo) { "1" } else { "0" }
& $py -c "import sys, importlib.util as u; ok = bool(u.find_spec('torch') and u.find_spec('torchvision')) and (sys.argv[1] == '0' or __import__('torch').cuda.is_available()); sys.exit(0 if ok else 1)" $needCuda
if ($LASTEXITCODE -eq 0) {
    Step "Το PyTorch είναι ήδη εγκατεστημένο και βλέπει την κάρτα: παραλείπεται." "PyTorch is already installed and working: skipped."
} else {
    Step "Εγκατάσταση PyTorch για την κάρτα σου (μπορεί να πάρει λίγα λεπτά)..." "Installing PyTorch for your GPU (can take a few minutes)..."
    Run $uv @("pip", "install", "--python", $py, "torch", "torchvision", "--torch-backend=auto")
}

Step "Εγκατάσταση της εφαρμογής..." "Installing the app..."
$extras = if ($gpuInfo) { "quant,desktop" } else { "desktop" }
Run $uv @("pip", "install", "--python", $py, "-e", ".[$extras]", "onnxruntime-directml", "--torch-backend=auto")

Step "Έλεγχος..." "Checking..."
& $py -c "import torch; ok = torch.cuda.is_available(); print('    PyTorch', torch.__version__, '| CUDA:', ok, '|', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if ok else '')"
if ($gpuInfo) {
    & $py -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!  Το PyTorch δεν βλέπει την κάρτα NVIDIA. Ενημέρωσε τον driver της NVIDIA και ξανατρέξε το install.bat." -ForegroundColor Yellow
        Write-Host "   PyTorch can't see the NVIDIA GPU. Update the NVIDIA driver and run install.bat again." -ForegroundColor DarkGray
    }
}

# ---------------------------------------------------------------- shortcuts
Step "Δημιουργία συντομεύσεων..." "Creating shortcuts..."
$pyw = Join-Path $Root ".venv\Scripts\pythonw.exe"
$icon = Join-Path $Root "edit_studio\web\img\icon.ico"
$shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
    try {
        $lnk = $shell.CreateShortcut((Join-Path $dir "Edit Studio.lnk"))
        $lnk.TargetPath = $pyw
        $lnk.Arguments = "-m edit_studio"
        $lnk.WorkingDirectory = $Root
        $lnk.IconLocation = $icon
        $lnk.Description = "Edit Studio - AI images & video"
        $lnk.Save()
        Write-Host "    $dir\Edit Studio.lnk"
    } catch { Write-Host "    (skip $dir)" }
}

# ---------------------------------------------------------------- version
$version = ""
$commit = ""
try { $version = (& $py -c "import edit_studio; print(edit_studio.__version__)") } catch { }
if (Test-Path (Join-Path $Root ".git")) {
    try { $commit = (& git -C $Root log -1 --format="%h, %cd" --date=short) } catch { }
}
Write-Host ""
Write-Host "Edit Studio $version $(if ($commit) { "($commit)" })" -ForegroundColor Magenta
if (-not (Test-Path (Join-Path $Root ".git"))) {
    Write-Host "    Για νεότερη έκδοση: κατέβασε ξανά το ZIP από το GitHub, αντέγραψε τα αρχεία πάνω σε αυτόν τον φάκελο" -ForegroundColor DarkGray
    Write-Host "    και τρέξε το update.bat. / To update: download the ZIP again, copy it over this folder, run update.bat." -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "OK! Η εγκατάσταση ολοκληρώθηκε." -ForegroundColor Green
Write-Host "Άνοιξε το «Edit Studio» από την επιφάνεια εργασίας (ή τρέξε το EditStudio.bat)." -ForegroundColor Green
Write-Host "Done! Open 'Edit Studio' from your desktop (or run EditStudio.bat)." -ForegroundColor DarkGray
