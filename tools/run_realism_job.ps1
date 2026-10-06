param([Parameter(Mandatory=$true)][string]$JobPath)
$ErrorActionPreference='Stop'
$jobSpec=Get-Content -LiteralPath $JobPath -Raw -Encoding UTF8 | ConvertFrom-Json
$taskStatePath=Join-Path $jobSpec.output 'state.json'
$taskState=Get-Content -LiteralPath $taskStatePath -Raw -Encoding UTF8 | ConvertFrom-Json
$deadline=[DateTimeOffset]::Parse($jobSpec.deadline_utc)
$remaining=($deadline-[DateTimeOffset]::UtcNow).TotalMilliseconds
if($remaining -lt 60000){throw 'Deadline too close; refuse new work'}
$jobArgs=@('-b',('"'+$jobSpec.blend+'"'),'--python-exit-code','9','--python',('"'+$jobSpec.script+'"'),'--',('"'+$jobSpec.root+'"'))
$taskProcess=Start-Process -FilePath 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' -ArgumentList $jobArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $jobSpec.output ($jobSpec.id+'.stdout.log')) -RedirectStandardError (Join-Path $jobSpec.output ($jobSpec.id+'.stderr.log'))
$taskState.active_pid=$taskProcess.Id
$taskState.stage=$jobSpec.id+'_running'
$taskState | Add-Member -NotePropertyName active_job -NotePropertyValue $JobPath -Force
$taskState.updated_utc=[DateTime]::UtcNow.ToString('o')
[IO.File]::WriteAllText($taskStatePath,($taskState|ConvertTo-Json -Depth 20),[Text.UTF8Encoding]::new($false))
Write-Output ('Owned Blender PID '+$taskProcess.Id)
$completed=$taskProcess.WaitForExit([int][Math]::Min($remaining,1000*$jobSpec.timeout_seconds))
if(-not $completed){$taskProcess.Kill();$taskProcess.WaitForExit()}
$taskProcess.Refresh()
# The reviewer may have updated next_action/latest_review while rendering.
# Reload before updating only worker-owned status, rather than restoring stale metadata.
$taskState=Get-Content -LiteralPath $taskStatePath -Raw -Encoding UTF8 | ConvertFrom-Json
$taskState.active_pid=$null
$taskState.stage=$jobSpec.id+'_exited_needs_output_verification'
if(-not $completed){$taskState.stage=$jobSpec.id+'_timed_out'}
$taskState.updated_utc=[DateTime]::UtcNow.ToString('o')
[IO.File]::WriteAllText($taskStatePath,($taskState|ConvertTo-Json -Depth 20),[Text.UTF8Encoding]::new($false))
Write-Output $taskState.stage
