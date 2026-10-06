"""Read the downloaded release, without installing into existing environments."""
from pathlib import Path
import zipfile
import sys
p=Path(__file__).resolve().parents[1]/'output/teacher_temporal_cycle_20261001/dependency_probe/lightning_pose-2.4.2-py3-none-any.whl'
with zipfile.ZipFile(p) as z:
    if len(sys.argv)>1:
        for name in sys.argv[1:]:print('FILE',name,'\n',z.read(name).decode())
    else:
        for name in z.namelist():
            if name.endswith('.py') and any(x in name for x in ['train','data/','utils/scripts']):print(name)
