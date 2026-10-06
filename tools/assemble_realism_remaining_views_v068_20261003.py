import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
root = Path(__file__).resolve().parents[1]
out = root / 'output/pymousestudio_realism_cycle_20261002/v068_remaining_views'
manifest = json.loads((out / 'manifest.json').read_text())
assert manifest['complete'] and all(x['blend_unchanged'] for x in manifest['poses'])
sheet = Image.new('RGB', (1080, 1320), '#152131')
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype('/mnt/c/Windows/Fonts/arial.ttf', 24)
for i, pose in enumerate(['climb', 'quadruped', 'headraise']):
    for j, cam in enumerate('CEG'):
        p = out / pose / f'cam{cam}.png'
        im = Image.open(p)
        assert im.size == (1080, 1200)
        draw.text((j*360+10, i*440+5), f'{pose} cam{cam}', fill='white', font=font)
        sheet.paste(im.convert('RGB').resize((360, 400)), (j*360, i*440+40))
sheet.save(out / 'remaining_views_contact.jpg', quality=96)
print(out / 'remaining_views_contact.jpg')
