import runpy,sys
from pathlib import Path
root=Path(sys.argv[-1])
runpy.run_path(str(root/'BlenderScripts/probe_ear_materials_20261003.py'),init_globals={'VERSION':'v002','CANDIDATES':[('a_albedo045',{'gray':.45}),('b_albedo075',{'gray':.75}),('c_albedo060_sss',{'gray':.60,'sss':.20,'scale':.35}),('d_albedo060_trans',{'gray':.60,'translucency':.15})]},run_name='__main__')
