import openslide
import os

folder_path = 'C:/Users/65202/Desktop/train'#svs所在路径
out_path = 'C:/Users/65202/Desktop/TEST'#输出路径
def seg(slide_path):
    slide = openslide.open_slide(slide_path)#读取
    N = 512
    [m, n] = slide.dimensions
    [ml, nl] = [m // N * N, n // N * N] #忽略边缘
    for i in range(0, ml, N):
        for j in range(0, nl, N):
            tile = slide.read_region((i, j), 0, (N, N))#读取小块
            tile_rgb = tile.convert('RGB')  # RGBA转为RGB
            sub_path = os.path.join(out_path, file_name)#out_path下子文件夹的名称
            os.makedirs(sub_path,exist_ok=True)#建立该子文件夹
            tile_rgb.save(sub_path + '/' +file_name + '(' + str(int(i / N)) + ',' + str(int(j / N)) + ')' + '.jpg')
    slide.close()

for root,dirs,files in os.walk(folder_path):
    for file in files:
        file_name = file[:-4]#去除.svs后缀
        slide_path = os.path.join(folder_path, file)#每个SVS文件的绝对路径
        seg(slide_path)#遍历每个SVS文件，分割

