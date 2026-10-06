"""Scientific contact sheets: all CAL bottom rows, GT and regional candidates."""
import json,shutil
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/local_grouping_cycle_20261005/peak_integrity_v1'
DEST=Path('/mnt/c/Users/55395/Downloads/PyMouseStudio-LocalGrouping-20261005/peak_integrity_v1')
def main():
    examples=json.loads((OUT/'examples.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for arm in ['native_rgb','quarter_turn']:
        rows=[r for r in examples if r['arm']==arm and '_calibration_real_' in r['file']]
        for start in range(0,len(rows),4):
            tile=Image.new('RGB',(1320,1580),'white');d=ImageDraw.Draw(tile)
            for slot,r in enumerate(rows[start:start+4]):
                im=Image.open(OUT/r['file']);w=im.width//3;h=im.height-65
                a=im.crop((0,35,w,35+h));b=im.crop((w*2,35,w*3,35+h));pair=Image.new('RGB',(w*2,h));pair.paste(a,(0,0));pair.paste(b,(w,0));pair.thumbnail((650,720))
                y,x=divmod(slot,2);tile.paste(pair,(x*660,y*790+40));d.text((x*660+5,y*790+5),f'{arm} {r["index"]}: GT | regional candidates',font=font,fill='black')
            name=f'{arm}_bottom_contact_{start//4+1}.jpg';tile.save(OUT/name,quality=94);shutil.copy2(OUT/name,DEST/name)
    print('All CAL bottom contact sheets exported')
if __name__=='__main__':main()
