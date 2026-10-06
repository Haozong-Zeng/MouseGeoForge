import sys
from pathlib import Path
root=Path(sys.argv[-1]);OUTPUT_VERSION='final_balanced';DELIVERY_FOLDER='DELIVERY-Balanced';DUST_FACTOR=.02
exec(compile((root/'BlenderScripts/fix_bottom_tail_shadow_20261003.py').read_text(),'fix_bottom_tail_shadow_20261003.py','exec'))
