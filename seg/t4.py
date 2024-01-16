import openslide
from openslide import deepzoom
import matplotlib.pyplot as plt
with openslide.OpenSlide(r"D:\dataset\cancer\C2023-11-02 01_16_39.svs") as slide:
    thumbnail = slide.get_thumbnail((256, 256))
    plt.imshow(thumbnail)
plt.show()