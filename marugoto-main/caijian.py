from PIL import Image
import os

def crop_and_save(image_path, output_folder, crop_size=224):
    # 打开图像
    img = Image.open(image_path)

    # 获取原图的名称（去除路径和扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 创建保存小图的文件夹
    output_folder_path = os.path.join(output_folder, image_name)
    os.makedirs(output_folder_path, exist_ok=True)

    # 获取原图尺寸
    img_width, img_height = img.size

    # 计算裁剪小图的数量
    num_crops_x = img_width // crop_size
    num_crops_y = img_height // crop_size

    # 循环裁剪并保存小图
    for x in range(num_crops_x):
        for y in range(num_crops_y):
            # 计算裁剪区域的坐标
            left = x * crop_size
            upper = y * crop_size
            right = left + crop_size
            lower = upper + crop_size

            # 裁剪图像
            cropped_img = img.crop((left, upper, right, lower))

            # 计算绝对坐标
            absolute_x = x * crop_size
            absolute_y = y * crop_size

            # 构建保存小图的文件名，包括绝对坐标
            crop_name = f"{image_name}({absolute_x},{absolute_y}).jpg"
            crop_path = os.path.join(output_folder_path, crop_name)

            # 保存小图
            cropped_img.save(crop_path)

if __name__ == "__main__":
    # 输入图像路径
    input_image_path = r"C:\Users\65202\Desktop\test\tree.jpg"

    # 输出文件夹路径
    output_folder_path = r"C:\Users\65202\Desktop\test"

    # 裁剪并保存小图
    crop_and_save(input_image_path, output_folder_path)
