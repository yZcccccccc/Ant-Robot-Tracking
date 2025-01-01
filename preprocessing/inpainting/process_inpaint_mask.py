import os
import numpy as np
from PIL import Image, ImageFilter

# 输入文件夹路径
input_folder = "D:/result/mobile/wx_mobile/fin"
# 输出文件夹路径
output_folder = "D:/result/240608/wx"

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 获取文件夹中所有图片文件
image_files = [f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')]

# 处理每张图片
for image_file in image_files:
    # 读取图像
    image_path = os.path.join(input_folder, image_file)
    image = Image.open(image_path).convert("L")
    image_data = np.array(image)

    # 计算白色像素占比
    total_pixels = image_data.size
    white_pixels = np.sum(image_data == 255)
    white_pixel_percentage = (white_pixels / total_pixels) * 100

    # 如果白色像素占比超过10%，进行处理
    while white_pixel_percentage > 10:
        # 对图像进行腐蚀操作（缩小白色区域）
        image = Image.fromarray(image_data)
        image = image.filter(ImageFilter.MinFilter(3))  # 3x3最小滤波器，类似于腐蚀操作
        image_data = np.array(image)

        # 更新白色像素占比
        white_pixels = np.sum(image_data == 255)
        white_pixel_percentage = (white_pixels / total_pixels) * 100

    # 将修改后的图像保存到输出文件夹
    output_image_path = os.path.join(output_folder, image_file)
    output_image = Image.fromarray(image_data)
    output_image.save(output_image_path)

print("处理完成！")
