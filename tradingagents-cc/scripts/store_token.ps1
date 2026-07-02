# Stores CLAUDE_CODE_OAUTH_TOKEN into Windows Credential Manager (generic
# credential 'tradingagents-cc/CLAUDE_CODE_OAUTH_TOKEN') via the raw Win32
# CredWriteW API — the exact mirror of run_daily.ps1's CredReadW fallback, so
# it works without the CredentialManager module and without cmdkey (whose
# command-line parsing chokes on some pasted secrets).
#
# Usage (interactive PowerShell window):
#   .\store_token.ps1          # prompts for the token (masked), stores, verifies
#
# The token never appears on a process command line. Whitespace is trimmed
# before storage (a leading space in a pasted token still authenticates, but
# has caused confusion before).

$ErrorActionPreference = 'Stop'
$CredTarget = 'tradingagents-cc/CLAUDE_CODE_OAUTH_TOKEN'

$secure = Read-Host -Prompt 'Paste CLAUDE_CODE_OAUTH_TOKEN' -AsSecureString
$bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try { $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }

$token = $token.Trim()
if ([string]::IsNullOrEmpty($token)) {
    Write-Host 'ERROR: empty token; nothing stored.' -ForegroundColor Red
    exit 1
}
if ($token -notlike 'sk-ant-*') {
    Write-Host "WARNING: token does not start with 'sk-ant-'; storing anyway." -ForegroundColor Yellow
}

if (-not ('TradingAgentsCC.NativeCredWrite' -as [type])) {
    Add-Type -Namespace TradingAgentsCC -Name NativeCredWrite -MemberDefinition @'
[DllImport("advapi32.dll", EntryPoint = "CredWriteW", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern bool CredWrite(ref CREDENTIAL credential, int flags);

[DllImport("advapi32.dll", EntryPoint = "CredReadW", CharSet = CharSet.Unicode, SetLastError = true)]
public static extern bool CredRead(string target, int type, int flags, out IntPtr credPtr);

[DllImport("advapi32.dll")]
public static extern void CredFree(IntPtr credPtr);

[StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
public struct CREDENTIAL
{
    public int Flags;
    public int Type;
    [MarshalAs(UnmanagedType.LPWStr)] public string TargetName;
    [MarshalAs(UnmanagedType.LPWStr)] public string Comment;
    public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
    public int CredentialBlobSize;
    public IntPtr CredentialBlob;
    public int Persist;
    public int AttributeCount;
    public IntPtr Attributes;
    [MarshalAs(UnmanagedType.LPWStr)] public string TargetAlias;
    [MarshalAs(UnmanagedType.LPWStr)] public string UserName;
}
'@
}

$CRED_TYPE_GENERIC = 1
$CRED_PERSIST_LOCAL_MACHINE = 2

$blob = [Runtime.InteropServices.Marshal]::StringToCoTaskMemUni($token)
try {
    $cred = New-Object TradingAgentsCC.NativeCredWrite+CREDENTIAL
    $cred.Type = $CRED_TYPE_GENERIC
    $cred.TargetName = $CredTarget
    $cred.UserName = 'CLAUDE_CODE_OAUTH_TOKEN'
    $cred.CredentialBlob = $blob
    $cred.CredentialBlobSize = $token.Length * 2  # UTF-16 bytes
    $cred.Persist = $CRED_PERSIST_LOCAL_MACHINE

    if (-not [TradingAgentsCC.NativeCredWrite]::CredWrite([ref]$cred, 0)) {
        $err = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        Write-Host "ERROR: CredWrite failed (Win32 error $err)." -ForegroundColor Red
        exit 1
    }
} finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeCoTaskMemUnicode($blob)
}
$token = $null

# Verify round-trip with the same CredRead path run_daily.ps1 uses.
$ptr = [IntPtr]::Zero
if ([TradingAgentsCC.NativeCredWrite]::CredRead($CredTarget, $CRED_TYPE_GENERIC, 0, [ref]$ptr)) {
    try {
        $c = [Runtime.InteropServices.Marshal]::PtrToStructure($ptr, [type]'TradingAgentsCC.NativeCredWrite+CREDENTIAL')
        $len = [int]($c.CredentialBlobSize / 2)
        Write-Host "Stored and verified: $CredTarget ($len chars)." -ForegroundColor Green
    } finally {
        [TradingAgentsCC.NativeCredWrite]::CredFree($ptr)
    }
} else {
    Write-Host 'ERROR: stored but read-back failed; run_daily.ps1 may not see it.' -ForegroundColor Red
    exit 1
}
