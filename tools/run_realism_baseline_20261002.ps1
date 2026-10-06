$ErrorActionPreference = 'Stop'
$taskRoot = '\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge'
$taskOutput = Join-Path $taskRoot 'output\pymousestudio_realism_cycle_20261002'
$taskStatePath = Join-Path $taskOutput 'state.json'
$taskState = Get-Content -LiteralPath $taskStatePath -Raw | ConvertFrom-Json
$taskArgs = @('-b', ('"' + $taskState.best_blend.Replace('/home/haozong/projects/MouseGeoForge', $taskRoot).Replace('/', '\') + '"'), '--python', ('"' + (Join-Path $taskRoot 'BlenderScripts\render_realism_baseline_20261002.py') + '"'), '--', ('"' + (Join-Path $taskOutput 'baseline_v000') + '"'))
$taskProcess = Start-Process -FilePath 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe' -ArgumentList $taskArgs -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'baseline.stdout.log') -RedirectStandardError (Join-Path $taskOutput 'baseline.stderr.log')
$taskState.active_pid = $taskProcess.Id
$taskState.stage = 'baseline_gpu_render_running'
$taskState.updated_utc = [DateTime]::UtcNow.ToString('o')
[IO.File]::WriteAllText($taskStatePath, ($taskState | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))
Write-Output ('Owned Blender PID ' + $taskProcess.Id)
if (-not $taskProcess.WaitForExit(900000)) {
    $taskProcess.Kill()
    $taskProcess.WaitForExit()
    $taskState.stage = 'baseline_timed_out_review_partial_results'
} else {
    $taskProcess.Refresh()
    if ($taskProcess.ExitCode -eq 0) { $taskState.stage = 'baseline_finished_needs_visual_review' }
    else { $taskState.stage = 'baseline_failed_check_logs' }
}
$taskState.active_pid = $null
$taskState.updated_utc = [DateTime]::UtcNow.ToString('o')
[IO.File]::WriteAllText($taskStatePath, ($taskState | ConvertTo-Json -Depth 20), [Text.UTF8Encoding]::new($false))
Write-Output $taskState.stage
