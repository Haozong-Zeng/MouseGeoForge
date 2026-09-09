param()

$ErrorActionPreference = 'Stop'

$projectRoot = '\\wsl.localhost\Ubuntu\home\haozong\projects\MouseGeoForge'
$projectName = 'Homecage-BlackBrownFur-LightningPose-CRIM13-DiversePose200-DepthHeightV1'
$blender = 'C:\Program Files\Blender Foundation\Blender 5.1\blender.exe'
$blend = Join-Path $projectRoot "BlenderPyMouseProjects\Homecage\$projectName.blend"
$dataset = Join-Path $projectRoot "dataset\BlenderPyMouse\$projectName"
$renderScript = Join-Path $projectRoot 'BlenderScripts\run_homecage_dual_mouse_diverse200_render.py'
$augmentScript = Join-Path $projectRoot 'BlenderScripts\augment_dual_mouse_blendedmvs_core23.py'
$validateScript = Join-Path $projectRoot 'BlenderScripts\validate_dual_homecage_blenderpymouse.py'
$sourcePlan = Join-Path $projectRoot 'dataset\Homecage\LightningPose-CRIM13-Diverse200.plan.json'
$statusPath = Join-Path $dataset 'pipeline_status.json'
$validationReport = Join-Path $dataset 'dataset_validation.json'
$logRoot = Join-Path $projectRoot 'dataset\BlenderPyMouse\_render_logs'
$renderStdout = Join-Path $logRoot "$projectName.stdout.log"
$renderStderr = Join-Path $logRoot "$projectName.stderr.log"
$augmentStdout = Join-Path $logRoot "$projectName.augment.stdout.log"
$augmentStderr = Join-Path $logRoot "$projectName.augment.stderr.log"
$validateStdout = Join-Path $logRoot "$projectName.validate.stdout.log"
$validateStderr = Join-Path $logRoot "$projectName.validate.stderr.log"

New-Item -ItemType Directory -Force -Path $dataset | Out-Null
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

function Write-PipelineStatus {
    param([string]$Stage, [hashtable]$Extra = @{})
    $payload = [ordered]@{
        stage = $Stage
        updated_at_utc = [DateTimeOffset]::UtcNow.ToString('o')
        project_name = $projectName
        source_blend = $blend
        dataset_root = $dataset
        frame_range = @(1, 200)
        view_count = 4
        cameras = @(
            'HomecageCam_Top',
            'HomecageCam_SideNegX',
            'HomecageCam_SideNegY',
            'HomecageCam_SidePosY'
        )
        resolution = @(1280, 720)
        rgb_engine = 'CYCLES'
        cycles_samples = 64
        depth_wall_policy = 'side: ignore only nearest camera-facing wall; top: retain full shell'
        pose_or_animation_edits = $false
        background_supervisor_pid = $PID
    }
    foreach ($key in $Extra.Keys) {
        $payload[$key] = $Extra[$key]
    }
    $temporary = "$statusPath.tmp"
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

try {
    Write-PipelineStatus -Stage 'rendering_rgb_depth_background' -Extra @{
        render_stdout = $renderStdout
        render_stderr = $renderStderr
    }
    $render = Start-Process -FilePath $blender -ArgumentList @(
        '-b', $blend, '-P', $renderScript
    ) -WindowStyle Hidden -PassThru -Wait `
      -RedirectStandardOutput $renderStdout -RedirectStandardError $renderStderr
    if ($render.ExitCode -ne 0) {
        throw "RGB/depth Blender process failed with exit code $($render.ExitCode)"
    }

    Write-PipelineStatus -Stage 'augmenting_dual_instance_core23' -Extra @{
        render_exit_code = $render.ExitCode
        augment_stdout = $augmentStdout
        augment_stderr = $augmentStderr
    }
    $augment = Start-Process -FilePath $blender -ArgumentList @(
        '-b', $blend,
        '-P', $augmentScript,
        '--',
        '--dataset-root', $dataset,
        '--frame-start', '1',
        '--frame-end', '200',
        '--no-resume',
        '--source-plan', $sourcePlan
    ) -WindowStyle Hidden -PassThru -Wait `
      -RedirectStandardOutput $augmentStdout -RedirectStandardError $augmentStderr
    if ($augment.ExitCode -ne 0) {
        throw "Dual-instance augmenter failed with exit code $($augment.ExitCode)"
    }

    Write-PipelineStatus -Stage 'validating_dataset' -Extra @{
        render_exit_code = $render.ExitCode
        augment_exit_code = $augment.ExitCode
        validation_report = $validationReport
    }
    $validate = Start-Process -FilePath $blender -ArgumentList @(
        '-b', $blend,
        '-P', $validateScript,
        '--',
        $dataset,
        $validationReport
    ) -WindowStyle Hidden -PassThru -Wait `
      -RedirectStandardOutput $validateStdout -RedirectStandardError $validateStderr
    if ($validate.ExitCode -ne 0) {
        throw "Dataset validation failed with exit code $($validate.ExitCode)"
    }

    Write-PipelineStatus -Stage 'complete' -Extra @{
        render_exit_code = $render.ExitCode
        augment_exit_code = $augment.ExitCode
        validation_exit_code = $validate.ExitCode
        validation_report = $validationReport
        scene_count = 200
        image_count = 800
        depth_count = 800
        instance_mask_count = 1600
        core23_instance_count = 400
    }
}
catch {
    Write-PipelineStatus -Stage 'failed' -Extra @{
        error = $_.Exception.Message
        stack = $_.ScriptStackTrace
        render_stdout = $renderStdout
        render_stderr = $renderStderr
        augment_stdout = $augmentStdout
        augment_stderr = $augmentStderr
    }
    throw
}
