"""Bounded Windows worker under shared GPU lock; manifest verification is separate."""
import fcntl,subprocess,sys,json,datetime as dt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def win(p):return '\\\\wsl.localhost\\Ubuntu'+str(p.resolve()).replace('/','\\')
path=Path(sys.argv[1]).resolve();spec=json.loads(path.read_text())
assert dt.datetime.now(dt.timezone.utc)<dt.datetime.fromisoformat(spec['deadline_utc'])
with (ROOT/'output/teacher_temporal_cycle_20261001/gpu_job.lock').open('a+') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    subprocess.run(['/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',win(ROOT/'tools/run_realism_job.ps1'),'-JobPath',win(path)],check=True)
