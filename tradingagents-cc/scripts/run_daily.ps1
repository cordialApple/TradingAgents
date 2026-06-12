# tradingagents-cc Task Scheduler wrapper (Windows PowerShell 5.1 compatible).
# Resolves CLAUDE_CODE_OAUTH_TOKEN from Windows Credential Manager, sets
# PYTHONUTF8=1, runs `python -m tradingagents_cc.routine` via the workspace
# venv interpreter by absolute path (Task Scheduler activates no venv), tees
# all output to %LOCALAPPDATA%\tradingagents-cc\logs\routine_YYYYMMDD.log
# (Task Scheduler swallows stdout), and exits with the python exit code:
#   0 = ok or market-closed skip, 1 = partial (>=1 ticker failed),
#   2 = fatal/auth (remediation: claude setup-token, then register_task.ps1).

param(
    # Forwarded verbatim to the routine module
    # (register_task.ps1 -DryRun wires `--dry-run` through here).
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RoutineArgs
)

# Native stderr is merged into the pipeline below; 'Stop' would turn the first
# stderr line (python logging) into a terminating error in PowerShell 5.1.
$ErrorActionPreference = 'Continue'

$PythonExe   = 'C:\Users\randl\Documents\GitHub\TradingAgents\.venv\Scripts\python.exe'
$ProjectRoot = 'C:\Users\randl\Documents\GitHub\TradingAgents\tradingagents-cc'
$CredTarget  = 'tradingagents-cc/CLAUDE_CODE_OAUTH_TOKEN'

if ($null -eq $RoutineArgs) { $RoutineArgs = @() }

$LogDir  = Join-Path $env:LOCALAPPDATA 'tradingagents-cc\logs'
$LogFile = Join-Path $LogDir ('routine_{0}.log' -f (Get-Date -Format 'yyyyMMdd'))
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log {
    param([string]$Message)
    $line = '{0} [run_daily] {1}' -f (Get-Date -Format 'HH:mm:ss'), $Message
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
    Write-Host $line
}

# Python emits UTF-8 (PYTHONUTF8=1); decode it as such when teeing. There is
# no console handle under Task Scheduler, so failures here are ignorable.
try { [Console]::OutputEncoding = [Text.Encoding]::UTF8 } catch { }

# --- Token: Credential Manager first (module, then raw Win32 CredReadW so
# --- cmdkey-stored credentials work without the module), final fallback is
# --- whatever CLAUDE_CODE_OAUTH_TOKEN the process inherited (User scope).
$token = $null
if (Get-Module -ListAvailable -Name CredentialManager) {
    try {
        Import-Module CredentialManager -ErrorAction Stop
        $cred = Get-StoredCredential -Target $CredTarget -ErrorAction Stop
        if ($null -ne $cred) { $token = $cred.GetNetworkCredential().Password }
    } catch { $token = $null }
}
if ([string]::IsNullOrEmpty($token)) {
    try {
        if (-not ('TradingAgentsCC.NativeCred' -as [type])) {
            Add-Type -Namespace TradingAgentsCC -Name NativeCred -MemberDefinition @'
[DllImport("advapi32.dll", EntryPoint = "CredReadW", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern bool CredRead(string target, int type, int flags, out IntPtr credPtr);

[DllImport("advapi32.dll")]
public static extern void CredFree(IntPtr credPtr);

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
public struct CREDENTIAL
{
    public int Flags;
    public int Type;
    public string TargetName;
    public string Comment;
    public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
    public int CredentialBlobSize;
    public IntPtr CredentialBlob;
    public int Persist;
    public int AttributeCount;
    public IntPtr Attributes;
    public string TargetAlias;
    public string UserName;
}
'@
        }
        $ptr = [IntPtr]::Zero
        if ([TradingAgentsCC.NativeCred]::CredRead($CredTarget, 1, 0, [ref]$ptr)) {
            try {
                $c = [Runtime.InteropServices.Marshal]::PtrToStructure($ptr, [type]'TradingAgentsCC.NativeCred+CREDENTIAL')
                if ($c.CredentialBlobSize -gt 0) {
                    $token = [Runtime.InteropServices.Marshal]::PtrToStringUni($c.CredentialBlob, [int]($c.CredentialBlobSize / 2))
                }
            } finally {
                [TradingAgentsCC.NativeCred]::CredFree($ptr)
            }
        }
    } catch { $token = $null }
}
if (-not [string]::IsNullOrEmpty($token)) {
    $env:CLAUDE_CODE_OAUTH_TOKEN = $token
    Write-Log "CLAUDE_CODE_OAUTH_TOKEN loaded from Credential Manager ($CredTarget)."
} elseif (-not [string]::IsNullOrEmpty($env:CLAUDE_CODE_OAUTH_TOKEN)) {
    Write-Log 'CLAUDE_CODE_OAUTH_TOKEN taken from the inherited environment (User scope).'
} else {
    Write-Log ("WARNING: no token in Credential Manager ($CredTarget) or environment; " +
        'routine preflight will exit 2 (run: claude setup-token, then scripts\register_task.ps1).')
}
$token = $null

$env:PYTHONUTF8 = '1'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    Write-Log "FATAL: venv interpreter not found at $PythonExe (moved repo or recreated .venv?)."
    exit 2
}
if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    Write-Log "FATAL: project root not found at $ProjectRoot (moved repo?)."
    exit 2
}
Set-Location -LiteralPath $ProjectRoot

Write-Log ('starting: "{0}" -m tradingagents_cc.routine {1}' -f $PythonExe, ($RoutineArgs -join ' '))

$exitCode = 2
try {
    & $PythonExe -m tradingagents_cc.routine @RoutineArgs 2>&1 | ForEach-Object {
        if ($_ -is [System.Management.Automation.ErrorRecord]) { $line = $_.Exception.Message }
        else { $line = [string]$_ }
        Add-Content -Path $LogFile -Value $line -Encoding UTF8
        Write-Host $line
    }
    $exitCode = $LASTEXITCODE
} catch {
    Write-Log ('FATAL: failed to launch routine: {0}' -f $_.Exception.Message)
    $exitCode = 2
}
if ($null -eq $exitCode) { $exitCode = 2 }

Write-Log "routine finished with exit code $exitCode."
exit $exitCode
