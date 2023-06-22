import os
import shutil

# 数据文件夹路径
data_folder = 'rawdata/test'
# 卷标文件路径
label_file = 'faceDS'
# 输出文件夹路径
output_folder = 'sorted_data_age/test'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 读取卷标文件
with open(label_file, 'r') as f:
    lines = f.readlines()

# 遍历卷标文件的每一行
for line in lines:
    # 解析标签信息
    parts = line.strip().split()
    file_name = parts[0]  # 文件名
    sex = None



    # 遍历卷标行的每个部分，查找_sex部分
    for i in range(1, len(parts)):
        if parts[i] == '(_age':
            if i + 1 < len(parts):
                sex = parts[i + 1]
            break

    # 检查是否成功提取到性别信息
    if sex is None:
        print(f"无法提取卷标行 '{line}' 中的性别信息")
        continue

    # 创建_sex文件夹（如果不存在）
    sex_folder = os.path.join(output_folder, sex)
    os.makedirs(sex_folder, exist_ok=True)

    # 复制数据文件到_sex文件夹
    file_path = os.path.join(data_folder, file_name)
    output_path = os.path.join(sex_folder, file_name)
    if not os.path.isfile(file_path):
        print(f"文件 '{file_path}' 不存在，跳过处理")
        continue
    shutil.copy(file_path, output_path)