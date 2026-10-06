"""Single GPU lock wraps bounded Windows Blender baseline runner."""
import fcntl,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
with (ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    script='\\\\wsl.localhost\\Ubuntu'+str(ROOT/'tools/run_realism_baseline_20261002.ps1').replace('/','\\')
    subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',script],check=True)
