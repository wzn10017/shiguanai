import openslide
slide = openslide.open_slide(r"D:\dataset\TCGA\tumor\TCGA-R6-A6DN-01Z-00-DX1.F0F21659-8F9E-415A-A0A8-8EB27FF6D312.svs")
print(slide.level_dimensions)