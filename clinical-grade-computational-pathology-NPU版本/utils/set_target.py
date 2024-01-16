import os
import csv

# 指定要处理的文件夹路径
folder_path = r'D:\dataset\testing'  # 将此路径替换为包含文件的文件夹路径

# 指定CSV文件路径
csv_file = r'D:\dataset\targets_testing.csv'  # 指定你要创建的CSV文件名

# 获取文件夹中的所有文件名
file_names = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_names.append(file)

# 将文件名写入CSV文件
with open(csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    for file_name in file_names:
        csv_writer.writerow([file_name])

print(f'文件名已写入到 {csv_file}')
