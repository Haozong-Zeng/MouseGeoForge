param([Parameter(Mandatory=$true)][string]$Config,[Parameter(Mandatory=$true)][string]$JobToken,[int]$TimeoutSeconds=7200)
$ErrorActionPreference='Stop'
$workspaceRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$settings=Get-Content -LiteralPath (Join-Path $workspaceRoot $Config) -Raw | ConvertFrom-Json
if($settings.job_token -ne $JobToken){throw 'Guardian token mismatch'}
$outputPath=Join-Path $workspaceRoot $settings.output
$identityPath=Join-Path $outputPath 'render_windows_process.json'
$began=[DateTime]::UtcNow
$deadline=[DateTime]::Parse($settings.deadline_utc).ToUniversalTime()
while(-not (Test-Path -LiteralPath $identityPath)){
 if(([DateTime]::UtcNow-$began).TotalSeconds -gt 60){throw 'No owned Blender process registered'}
 Start-Sleep -Seconds 1
}
$identity=Get-Content -LiteralPath $identityPath -Raw | ConvertFrom-Json
if($identity.job_token -ne $JobToken){throw 'Wrong owned Blender identity'}
$stopped=$false
while($true){
 $current=Get-CimInstance Win32_Process -Filter "ProcessId = $($identity.pid)" -ErrorAction SilentlyContinue
 if(-not $current){break}
 $same=$current.CreationDate.ToUniversalTime().ToString('o') -eq $identity.creation_utc -and $current.CommandLine -eq $identity.command -and $current.CommandLine.Contains($JobToken)
 if(-not $same){throw 'Blender PID reused; guardian will not stop it'}
 if([DateTime]::UtcNow -ge $deadline -or ([DateTime]::UtcNow-$began).TotalSeconds -ge $TimeoutSeconds -or (Test-Path -LiteralPath (Join-Path $outputPath 'STOP'))){
  Stop-Process -Id $identity.pid -Force
  $stopped=$true
  break
 }
 Start-Sleep -Seconds 2
}
@{finished_utc=[DateTime]::UtcNow.ToString('o');stopped_at_cutoff=$stopped;job_token=$JobToken;owned_pid=$identity.pid} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $outputPath 'windows_guardian_completion.json') -Encoding UTF8
