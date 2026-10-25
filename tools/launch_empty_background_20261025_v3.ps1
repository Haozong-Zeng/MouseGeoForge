param([Parameter(Mandatory=$true)][string]$Config,[Parameter(Mandatory=$true)][string]$JobToken,[ValidateSet('inspect','render')][string]$Command='inspect',[int]$TimeoutSeconds=600)
$ErrorActionPreference='Stop'
$workspaceRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
$configPath=Join-Path $workspaceRoot $Config
$settings=Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if($settings.job_token -ne $JobToken){throw 'Job token mismatch'}
$outputPath=Join-Path $workspaceRoot $settings.output
$binaryPath='C:\Program Files\Blender Foundation\Blender 5.1\blender.exe'
$scriptPath=Join-Path $workspaceRoot $settings.blender_script
$stdoutPath=Join-Path $outputPath ($Command+'_stdout.log')
$stderrPath=Join-Path $outputPath ($Command+'_stderr.log')
if(Test-Path -LiteralPath $stdoutPath){throw 'Existing render/inspect attempt is preserved'}
$arguments=@('--background','--disable-autoexec','--threads','8','--python-exit-code','1','--python',('"'+$scriptPath+'"'),'--','--config',$Config,'--job-token',$JobToken,'--command',$Command)
$child=Start-Process -FilePath $binaryPath -ArgumentList $arguments -WorkingDirectory $workspaceRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$null=$child.Handle
$childInfo=Get-CimInstance Win32_Process -Filter "ProcessId = $($child.Id)"
if(-not $childInfo -or -not $childInfo.CommandLine.Contains($JobToken)){throw 'Cannot establish exact Blender identity'}
$savedIdentity=@{pid=$child.Id;creation_utc=$childInfo.CreationDate.ToUniversalTime().ToString('o');command=$childInfo.CommandLine;executable=$childInfo.ExecutablePath;job_token=$JobToken}
$savedIdentity | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outputPath ($Command+'_windows_process.json')) -Encoding UTF8
$began=[DateTime]::UtcNow
$deadline=[DateTime]::Parse($settings.deadline_utc).ToUniversalTime()
try{
 while(-not $child.WaitForExit(2000)){
  if([DateTime]::UtcNow -ge $deadline -or ([DateTime]::UtcNow-$began).TotalSeconds -ge $TimeoutSeconds -or (Test-Path -LiteralPath (Join-Path $outputPath 'STOP'))){throw 'Finite Blender CPU diagnostic cutoff'}
 }
 if($child.ExitCode -ne 0){throw "Blender failed with exit code $($child.ExitCode)"}
}finally{
 $current=Get-CimInstance Win32_Process -Filter "ProcessId = $($child.Id)" -ErrorAction SilentlyContinue
 if($current -and $current.CreationDate.ToUniversalTime().ToString('o') -eq $savedIdentity.creation_utc -and $current.CommandLine -eq $savedIdentity.command -and $current.CommandLine.Contains($JobToken)){
  Stop-Process -Id $child.Id -Force
 }
}
