from pathlib import Path
import zipfile,sys
p=Path(__file__).resolve().parents[1]/'output/teacher_temporal_cycle_20261001/dependency_probe/ensemble_kalman_smoother-4.6.2-py3-none-any.whl'
with zipfile.ZipFile(p) as z:
    if len(sys.argv)>1:
        for name in sys.argv[1:]:print('FILE',name,'\n',z.read(name).decode())
    else:
        for name in z.namelist():
            if name.endswith('.py'):print(name)
            if name.endswith('METADATA'):
                print('\n'.join(l for l in z.read(name).decode().splitlines() if l.startswith(('Requires-','Version:'))))
