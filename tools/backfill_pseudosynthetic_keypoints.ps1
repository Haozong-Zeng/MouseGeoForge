[CmdletBinding()]
param(
    [string]$BlenderExe = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$annotationScript = Join-Path $projectRoot "BlenderScripts\annotate_existing_blenderpymouse_keypoints.py"
$datasetRoot = Join-Path $projectRoot "dataset\BlenderPyMouse"
$projectBase = Join-Path $projectRoot "BlenderPyMouseProjects\PseudoSynthetic"

if (-not (Test-Path -LiteralPath $BlenderExe -PathType Leaf)) {
    throw "Blender executable not found: $BlenderExe"
}
if (-not (Test-Path -LiteralPath $annotationScript -PathType Leaf)) {
    throw "Annotation script not found: $annotationScript"
}

$trials = @(
    "E_L2_2024-11-18_1_Trial_1",
    "FB_R3_2024-03-16_1_Trial_5",
    "FJ_L2_2024-06-24_1_Trial_7",
    "FS_L2_2024-06-25_1_Trial_2"
)
$datasetSuffixes = @("_PseudoSynthetic", "_PseudoSyntheticTransfered")

foreach ($trial in $trials) {
    $sourceDirectory = Join-Path $projectBase $trial
    $sourceBlend = Join-Path $sourceDirectory "${trial}_AniposeSyntheticReady.blend"
    $vertexMap = Join-Path $sourceDirectory "mouse_hand_vertex_keypoint_map.pkl"
    $vertexAudit = Join-Path $sourceDirectory "mouse_hand_vertex_keypoint_map.audit.json"

    foreach ($requiredPath in @($sourceBlend, $vertexMap, $vertexAudit)) {
        if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
            throw "Required source file not found: $requiredPath"
        }
    }

    foreach ($suffix in $datasetSuffixes) {
        $target = Join-Path $datasetRoot "${trial}${suffix}"
        if (-not (Test-Path -LiteralPath $target -PathType Container)) {
            throw "Dataset directory not found: $target"
        }

        $arguments = @(
            $sourceBlend,
            "--background",
            "--python", $annotationScript,
            "--",
            "--dataset-root", $target,
            "--vertex-keypoint-map", $vertexMap,
            "--vertex-keypoint-audit", $vertexAudit,
            "--progress-every", "25"
        )
        if ($Overwrite) {
            $arguments += "--overwrite"
        }

        Write-Host "Backfilling $($trial + $suffix)"
        & $BlenderExe @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Blender annotation failed for $target (exit $LASTEXITCODE)"
        }
    }
}

Write-Host "PseudoSynthetic keypoint backfill complete."
