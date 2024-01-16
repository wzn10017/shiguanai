from histolab.slide import Slide
from histolab.tiler import GridTiler
from histolab.tiler import RandomTiler
from PIL import Image
from histolab.masks import TissueMask

slide = Slide(r"D:\dataset\cancer\C2023-11-02 01_16_39.svs",r'D:\dataset')

all_tissue_mask = TissueMask()

grid_tiles_extractor = GridTiler(
   tile_size=(512, 512),
   level=0,
   check_tissue=False,
   pixel_overlap=0, # default
   prefix="grid/", # save tiles in the "grid" subdirectory of slide's processed_path
   suffix=".png" # default
)
grid_tiles_extractor.locate_tiles(
    slide=slide,
    extraction_mask=all_tissue_mask,
    scale_factor=64,
    alpha=64,
    outline="#046C4C",
).show()

grid_tiles_extractor.extract(slide,extraction_mask=all_tissue_mask)