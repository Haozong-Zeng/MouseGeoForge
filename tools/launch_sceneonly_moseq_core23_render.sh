#!/usr/bin/env bash
set -euo pipefail

workspace=/home/haozong/projects/MouseGeoForge
log_dir="$workspace/dataset/BlenderPyMouse/_render_logs"
stamp=$(date +%Y%m%d_%H%M%S)
stdout="$log_dir/sceneonly_moseq_core23_${stamp}.stdout.log"
stderr="$log_dir/sceneonly_moseq_core23_${stamp}.stderr.log"
blender="/mnt/c/Program Files/Blender Foundation/Blender 5.1/blender.exe"
blend=$(wslpath -w "$workspace/dataset/Cylinder/PyMouseStudio-fur-SceneOnly-MoSeqPose-Sample10-First200-mm.blend")
script=$(wslpath -w "$workspace/blender/run_sceneonly_moseq_core23_full_pipeline.py")

mkdir -p "$log_dir"
nohup "$blender" --background "$blend" --python "$script" \
    >"$stdout" 2>"$stderr" </dev/null &

echo "WSL_PID=$!"
echo "STDOUT=$stdout"
echo "STDERR=$stderr"
