# tradingagents-cc scheduler registration (Windows PowerShell 5.1 compatible).
# Stores the subscription token, then registers the Mon-Fri Task Scheduler job
# that runs scripts\run_daily.ps1 (which invokes python -m tradingagents_cc.routine).
#
# Usage (interactive PowerShell prompt; one-time prereq: claude setup-token):
#   .\register_task.ps1                   # store token + Mon-Fri 18:30 task 'TradingAgentsCC'
#   .\register_task.ps1 -Time 19:00       # custom local run time (24-hour HH:mm)
#   .\register_task.ps1 -DryRun           # one-shot task running `routine --dry-run` now
#                                         # (mock backend: zero credit, validates wiring only)
#   .\register_task.ps1 -PersistUserEnv   # store token as a User-scope env var instead
#
# Token storage (default): Windows Credential Manager generic credential
# 'tradingagents-cc/CLAUDE_CODE_OAUTH_TOKEN', via either write path — both are
# read back by run_daily.ps1:
#   1. CredentialManager module (Install-Module CredentialManager -Scope CurrentUser):
#      preferred; the token never appears on a process command line.
#   2. cmdkey fallback: works everywhere, but the token transits the cmdkey
#      command line for an instant — prefer 1 on shared machines.
# -PersistUserEnv instead writes plaintext to HKCU\Environment (simplest wiring,
# no module needed). Either way the secret is per-Windows-account: store it as
# the same account the scheduled task runs as.

param(
    [string]$Time = '18:30',
    [string]$TaskName = 'TradingAgentsCC',
    [switch]$DryRun,
    [switch]$PersistUserEnv
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = 'C:\Users\randl\Documents\GitHub\TradingAgents\tradingagents-cc'
$RunDaily    = Join-Path $ProjectRoot 'scripts\run_daily.ps1'
$PythonExe   = 'C:\Users\randl\Documents\GitHub\TradingAgents\.venv\Scripts\python.exe'
$CredTarget  = 'tradingagents-cc/CLAUDE_CODE_OAUTH_TOKEN'
$LogDir      = Join-Path $env:LOCALAPPDATA 'tradingagents-cc\logs'

if (-not (Test-Path -LiteralPath $RunDaily)) {
    Write-Host "ERROR: run_daily.ps1 not found at $RunDaily (moved repo?)." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Host "WARNING: venv interpreter not found at $PythonExe;" -ForegroundColor Yellow
    Write-Host "         run 'uv sync' at the workspace root before the first scheduled run." -ForegroundColor Yellow
}
try {
    $runAt = [datetime]::ParseExact($Time, 'HH:mm', [Globalization.CultureInfo]::InvariantCulture)
} catch {
    Write-Host "ERROR: -Time must be 24-hour HH:mm, e.g. 18:30 (got '$Time')." -ForegroundColor Red
    exit 1
}

# --- Token storage ----------------------------------------------------------
Write-Host "Token setup - generate one with: claude setup-token  (interactive, one-time)."
$secure = Read-Host -Prompt 'Paste CLAUDE_CODE_OAUTH_TOKEN (Enter to skip if already stored)' -AsSecureString
if ($secure.Length -gt 0) {
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try { $plainToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }

    if ($PersistUserEnv) {
        [Environment]::SetEnvironmentVariable('CLAUDE_CODE_OAUTH_TOKEN', $plainToken, 'User')
        Write-Host 'Token stored as User-scope environment variable CLAUDE_CODE_OAUTH_TOKEN.'
        Write-Host 'NOTE: User env vars are plaintext under HKCU\Environment; the Credential Manager default avoids that.'
    } elseif (Get-Module -ListAvailable -Name CredentialManager) {
        Import-Module CredentialManager
        New-StoredCredential -Target $CredTarget -UserName 'CLAUDE_CODE_OAUTH_TOKEN' `
            -Password $plainToken -Type Generic -Persist LocalMachine | Out-Null
        Write-Host "Token stored in Windows Credential Manager ($CredTarget) via the CredentialManager module."
    } else {
        & cmdkey /generic:$CredTarget /user:CLAUDE_CODE_OAUTH_TOKEN "/pass:$plainToken" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'ERROR: cmdkey failed to store the credential.' -ForegroundColor Red
            exit 1
        }
        Write-Host "Token stored in Windows Credential Manager ($CredTarget) via cmdkey."
        Write-Host 'NOTE: cmdkey exposes the token on its command line for an instant; for the next'
        Write-Host '      rotation, Install-Module CredentialManager -Scope CurrentUser avoids that.'
    }
    $plainToken = $null
} else {
    $userEnvToken = [Environment]::GetEnvironmentVariable('CLAUDE_CODE_OAUTH_TOKEN', 'User')
    $inCredMan = (cmdkey /list | Out-String) -match [regex]::Escape($CredTarget)
    if (-not $inCredMan -and [string]::IsNullOrEmpty($userEnvToken)) {
        Write-Host 'WARNING: no stored token found (Credential Manager or User-scope env);' -ForegroundColor Yellow
        Write-Host '         scheduled runs will fail preflight (exit 2) until one is stored.' -ForegroundColor Yellow
    } else {
        Write-Host 'Skipped - an existing stored token was found.'
    }
}

# --- Scheduled task ---------------------------------------------------------
$effectiveTaskName = $TaskName
$scriptArgs = '-NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $RunDaily
if ($DryRun) {
    $effectiveTaskName = "$TaskName-DryRun"
    $scriptArgs = "$scriptArgs --dry-run"
    $trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1)
    # EndBoundary + DeleteExpiredTaskAfter make the one-shot task self-clean.
    $trigger.EndBoundary = (Get-Date).AddHours(2).ToString('yyyy-MM-ddTHH:mm:ss')
    $description = 'tradingagents-cc wiring check: runs the routine once with --dry-run ' +
        '(mock backend, zero subscription credit), then self-deletes.'
} else {
    $trigger = New-ScheduledTaskTrigger -Weekly `
        -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $runAt
    $description = "tradingagents-cc daily routine (Mon-Fri $Time local, after US close): " +
        'unattended multi-agent trading pipeline via run_daily.ps1.'
}

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument $scriptArgs -WorkingDirectory $ProjectRoot

$settingsParams = @{
    AllowStartIfOnBatteries    = $true
    DontStopIfGoingOnBatteries = $true
    StartWhenAvailable         = $true
    MultipleInstances          = 'IgnoreNew'
    ExecutionTimeLimit         = (New-TimeSpan -Hours 2)
    RestartCount               = 2
    RestartInterval            = (New-TimeSpan -Minutes 30)
}
if ($DryRun) { $settingsParams['DeleteExpiredTaskAfter'] = (New-TimeSpan -Hours 1) }
$settings = New-ScheduledTaskSettingsSet @settingsParams

Register-ScheduledTask -TaskName $effectiveTaskName -Action $action -Trigger $trigger `
    -Settings $settings -Description $description -Force | Out-Null

# --- Verification + unattended-logon guidance -------------------------------
Write-Host ''
Write-Host "Registered scheduled task '$effectiveTaskName'." -ForegroundColor Green
if ($DryRun) {
    Write-Host 'Dry-run fires once within ~1 minute and self-deletes about an hour after expiry.'
} else {
    Write-Host "Runs Mon-Fri at $Time local time."
}
$logExample = Join-Path $LogDir ('routine_{0}.log' -f (Get-Date -Format 'yyyyMMdd'))
Write-Host ''
Write-Host 'Verify:'
Write-Host "  Start-ScheduledTask -TaskName '$effectiveTaskName'"
Write-Host "  Get-ScheduledTaskInfo -TaskName '$effectiveTaskName' | Format-List LastRunTime, LastTaskResult, NextRunTime"
Write-Host "  Get-Content `"$logExample`" -Tail 40   # date-stamped per run"
Write-Host '  LastTaskResult: 0 = ok or market-closed skip, 1 = partial (a ticker failed),'
Write-Host '                  2 = fatal/auth (remediation: claude setup-token, then re-run this script).'
Write-Host ''
Write-Host 'Run whether you are logged on or not (recommended for an evening schedule):'
Write-Host "  taskschd.msc -> '$effectiveTaskName' -> Properties -> General ->"
Write-Host "  select 'Run whether user is logged on or not' and enter your Windows password."
Write-Host '  Keep "Do not store password" UNCHECKED: a no-password (S4U) session cannot decrypt'
Write-Host '  Credential Manager secrets (DPAPI), so the stored token would be unreadable.'
Write-Host '  Equivalent CLI (prompts for your Windows password):'
Write-Host "  Set-ScheduledTask -TaskName '$effectiveTaskName' -User `$env:USERNAME -Password (Get-Credential `$env:USERNAME).GetNetworkCredential().Password"
exit 0
