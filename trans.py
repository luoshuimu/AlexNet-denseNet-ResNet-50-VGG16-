import os
import numpy as np
from PIL import Image

data_folder = 'rawdata_2'  # 二进制图像文件夹路径
output_folder = 'converted_images_2'  # 转换后的图像输出文件夹路径

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍历二进制图像文件夹中的文件
for filename in os.listdir(data_folder):
    file_path = os.path.join(data_folder, filename)

    # 读取二进制图像数据
    with open(file_path, 'rb') as f:
        binary_data = np.fromfile(f, dtype=np.uint8)

    # 将二进制数据转换为NumPy数组
    #image_array = np.frombuffer(binary_data, dtype=np.uint8)
    image = np.reshape(binary_data, (128, 128))
    # 将一维数组重新形状为图像尺寸
    #image = image_array.reshape((128, 128, 1))  # 根据实际图像尺寸进行调整

    # 创建PIL图像对象
    pil_image = Image.fromarray(image)

    # 构造输出文件路径
    output_path = os.path.join(output_folder, filename+'.jpg')  # 根据实际需求选择图像格式

    # 保存图像文件
    pil_image.save(output_path)