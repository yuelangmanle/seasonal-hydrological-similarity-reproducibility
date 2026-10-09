from pathlib import Path
from PIL import Image
root=Path(__file__).resolve().parent / "redrawn_figures"
# Native output width is 178 mm. Create width-labelled raster variants at 1000 dpi.
for stem in ['Figure2_internal_comparison','Figure3_regional_robustness','Figure1_diagnosis_construction']:
    src=root/(stem+'.png')
    im=Image.open(src).convert('RGB')
    for mm in [86,120,178]:
        px=round(mm/25.4*1000)
        ratio=px/im.width
        out=im.resize((px, round(im.height*ratio)), Image.Resampling.LANCZOS)
        out.save(root/f'{stem}_{mm}mm.png', dpi=(1000,1000))
        out.save(root/f'{stem}_{mm}mm.tiff', compression='tiff_lzw', dpi=(1000,1000))
