$ErrorActionPreference = "Stop"

$workspace = "\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge"
$logDir = Join-Path $workspace "dataset\BlenderPyMouse\_render_logs"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logDir "sceneonly_moseq_core23_${stamp}.stdout.log"
$stderr = Join-Path $logDir "sceneonly_moseq_core23_${stamp}.stderr.log"
$blend = Join-Path $workspace "BlenderPyMouseProjects\Cylinder\PyMouseStudio-fur-SceneOnly-MoSeqPose-Sample10-First200-mm.blend"
$script = Join-Path $workspace "BlenderScripts\run_sceneonly_moseq_core23_full_pipeline.py"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$process = Start-Process `
    -FilePath "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe" `
    -ArgumentList @("--background", $blend, "--python", $script) `
    -WindowStyle Hidden `
    -PassThru `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr

Write-Output "PID=$($process.Id)"
Write-Output "STDOUT=$stdout"
Write-Output "STDERR=$stderr"
