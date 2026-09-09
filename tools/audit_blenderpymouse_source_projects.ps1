[CmdletBinding()]
param(
    [string]$BlenderExe = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
    [string]$ReportPath = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$configurationPath = Join-Path $projectRoot "configs\blenderpymouse_production_projects.json"
$auditScript = Join-Path $projectRoot "BlenderScripts\audit_blend_external_dependencies.py"
if (-not $ReportPath) {
    $ReportPath = Join-Path $projectRoot "BlenderPyMouseProjects\source_dependency_audit.json"
}

$configuration = Get-Content -LiteralPath $configurationPath -Raw | ConvertFrom-Json
$rows = @()
foreach ($project in $configuration.projects) {
    $source = Join-Path $projectRoot $project.source
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        $destination = Join-Path (Join-Path $projectRoot $configuration.destination_root) $project.destination
        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            $source = $destination
        } else {
            throw "Production project not found at source or destination: $($project.source)"
        }
    }
    Write-Host "Auditing $source"
    $output = & $BlenderExe $source --background --python $auditScript 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Blender dependency audit failed for $source (exit $LASTEXITCODE)"
    }
    $payloadLine = $output | Where-Object { $_ -like "BLEND_DEPENDENCY_AUDIT=*" } | Select-Object -Last 1
    if (-not $payloadLine) {
        throw "Blender dependency audit produced no payload for $source"
    }
    $payload = $payloadLine.Substring("BLEND_DEPENDENCY_AUDIT=".Length) | ConvertFrom-Json
    $payload | Add-Member -NotePropertyName datasets -NotePropertyValue @($project.datasets)
    $rows += $payload
}

$report = [ordered]@{
    format = "mousegeoforge_blender_dependency_audit_v1"
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    project_count = $rows.Count
    unpacked_dependency_count = ($rows | Measure-Object -Property unpacked_dependency_count -Sum).Sum
    missing_unpacked_count = ($rows | Measure-Object -Property missing_unpacked_count -Sum).Sum
    projects = $rows
}
$reportDirectory = Split-Path -Parent $ReportPath
New-Item -ItemType Directory -Path $reportDirectory -Force | Out-Null
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $ReportPath -Encoding UTF8
[ordered]@{
    status = if ($report.missing_unpacked_count -eq 0) { "complete" } else { "failed" }
    project_count = $report.project_count
    unpacked_dependency_count = $report.unpacked_dependency_count
    missing_unpacked_count = $report.missing_unpacked_count
    report = $ReportPath
} | ConvertTo-Json
if ($report.missing_unpacked_count -ne 0) {
    exit 1
}
