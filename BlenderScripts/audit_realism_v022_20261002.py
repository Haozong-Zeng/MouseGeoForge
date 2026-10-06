import runpy,sys
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--')+1])
runpy.run_path(str(root/'BlenderScripts/audit_realism_groom_contact_20261002.py'),init_globals={'AUDIT_OUTPUT':'v022_constrained_groom/contact_audit.json','AUDIT_VERSIONS':['v020-b_chiang060','v022-a_normal_twist','v022-b_explicit_trim']})
