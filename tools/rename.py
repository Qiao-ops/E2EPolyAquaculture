# import os

# def rename_images(image_folder):
#     # 获取文件夹中所有文件的列表
#     image_files = sorted(os.listdir(image_folder))  # 排序文件名，以确保按字母顺序处理

#     # 遍历文件列表
#     for idx, file_name in enumerate(image_files):
#         # 检查是否为图片文件（根据文件扩展名）
#         if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
#             # 构造新的文件名，形式为 image_id.png（或其他扩展名）
#             file_extension = file_name.split('.')[-1]  # 获取文件扩展名
#             new_name = f"image_{idx}.{file_extension}"  # 新文件名，例如 0.png, 1.jpg 等

#             # 获取完整的文件路径
#             old_path = os.path.join(image_folder, file_name)
#             new_path = os.path.join(image_folder, new_name)

#             # 重命名文件
#             os.rename(old_path, new_path)
#             print(f"Renamed: {file_name} -> {new_name}")

# # 设置图像文件夹路径
# image_folder_path = '/qiaowenjiao/HiSup/data/yangzhiqu/lyg_3/cut_300/val/images'  # 请替换为实际的图像文件夹路径

# # 调用函数
# rename_images(image_folder_path)
import os
import shutil

def rename_images(image_folder, target_folder):
    # 获取文件夹中所有文件的列表
    image_files = sorted(os.listdir(image_folder))  # 排序文件名，以确保按字母顺序处理

    # 如果目标文件夹不存在，则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历文件列表
    for idx, file_name in enumerate(image_files):
        # 检查是否为图片文件（根据文件扩展名）
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 构造新的文件名，形式为 image_id.png（或其他扩展名）
            file_extension = file_name.split('.')[-1]  # 获取文件扩展名
            new_name = f"image_{idx}.{file_extension}"  # 新文件名，例如 image_0.png, image_1.jpg 等

            # 获取完整的文件路径
            old_path = os.path.join(image_folder, file_name)
            new_path = os.path.join(target_folder, new_name)

            # 将文件复制到目标文件夹并重命名
            shutil.copy(old_path, new_path)  # 使用 copy 而不是 rename，将文件复制到新文件夹
            print(f"Copied and renamed: {file_name} -> {new_name}")

# 设置图像文件夹路径和目标文件夹路径
image_folder_path = '/qiaowenjiao/HiSup/data/yangzhiqu/lyg_3/cut_300/val/images'  # 原图像文件夹路径
target_folder_path = '/qiaowenjiao/HiSup/data/yangzhiqu/lyg_3/cut_300/val/images_renamed'  # 目标文件夹路径

# 调用函数
rename_images(image_folder_path, target_folder_path)
