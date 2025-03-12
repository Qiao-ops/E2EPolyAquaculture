import arcpy
import os

# 输入 Shapefile 路径
input_polygon = "shp/shore.shp"

# 输入原始图像路径
input_tiff = "given.tif"

# 输出文件夹路径
output_folder_raster = r"E:\output_patches_raster"  # 原始图像的 patch
output_folder_polygon = r"E:\output_patches_polygon"  # 转换后的栅格的 patch

# Patch 大小（单位：像元）
patch_size = 512  # 512x512

# 重叠大小（单位：像元）
overlap_size = 128  # 指定重叠大小

# 选择用于栅格化的字段
field = "Id"

# 获取输入TIFF图像的像元大小
cell_size_x = arcpy.GetRasterProperties_management(input_tiff, "CELLSIZEX")[0]
cell_size_x = float(cell_size_x)
cell_size_y = arcpy.GetRasterProperties_management(input_tiff, "CELLSIZEY")[0]
cell_size_y = float(cell_size_y)

# 确保输出文件夹存在
for folder in [output_folder_raster, output_folder_polygon]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# 对输入TIFF图像进行重采样，将像元大小统一为X轴像元大小（结果保存在内存中）
resampled_tiff = arcpy.Resample_management(input_tiff, "in_memory/resampled_raster", cell_size_x, "NEAREST")

# 获取重采样后的栅格的空间范围和像元大小
desc_resampled = arcpy.Describe(resampled_tiff)
extent_resampled = desc_resampled.extent
cell_size_x_resampled = float(arcpy.GetRasterProperties_management(resampled_tiff, "CELLSIZEX")[0])
cell_size_y_resampled = float(arcpy.GetRasterProperties_management(resampled_tiff, "CELLSIZEY")[0])

# 获取重采样后的栅格的宽度和高度（以像元为单位）
width_resampled = int(arcpy.GetRasterProperties_management(resampled_tiff, "COLUMNCOUNT")[0])
height_resampled = int(arcpy.GetRasterProperties_management(resampled_tiff, "ROWCOUNT")[0])

# 设置环境范围与重采样后的栅格一致
arcpy.env.extent = extent_resampled

# 将面数据栅格化为栅格（结果保存在内存中）
polygon_raster = arcpy.PolygonToRaster_conversion(input_polygon, field, "in_memory/polygon_raster", cellsize=cell_size_x)

# 将栅格中的NoData值替换为0（结果保存在内存中）
output_raster_result = arcpy.sa.Con(arcpy.sa.IsNull(polygon_raster), 0, polygon_raster)

# 获取转换后的栅格的空间范围
desc_output_tiff = arcpy.Describe(output_raster_result)
extent_output_tiff = desc_output_tiff.extent

# 计算重叠区域
overlap_extent = arcpy.Extent(
    max(extent_resampled.XMin, extent_output_tiff.XMin),
    max(extent_resampled.YMin, extent_output_tiff.YMin),
    min(extent_resampled.XMax, extent_output_tiff.XMax),
    min(extent_resampled.YMax, extent_output_tiff.YMax)
)

# 计算重叠区域的宽度和高度（以像元为单位）
width_overlap = int((overlap_extent.XMax - overlap_extent.XMin) / cell_size_x_resampled)
height_overlap = int((overlap_extent.YMax - overlap_extent.YMin) / cell_size_y_resampled)

# 计算 patch 的数量（考虑重叠大小）
step_size = patch_size - overlap_size  # 步长
num_patches_x = (width_overlap - patch_size) // step_size + 1
num_patches_y = (height_overlap - patch_size) // step_size + 1

# 遍历每个 patch 并裁剪
for i in range(num_patches_x):
    for j in range(num_patches_y):
        # 计算当前 patch 的起始和结束位置
        xmin = overlap_extent.XMin + i * step_size * cell_size_x_resampled
        xmax = xmin + patch_size * cell_size_x_resampled
        ymin = overlap_extent.YMin + j * step_size * cell_size_y_resampled
        ymax = ymin + patch_size * cell_size_y_resampled

        # 定义当前 patch 的空间范围
        patch_extent = f"{xmin} {ymin} {xmax} {ymax}"

        # 裁剪原始图像
        patch_name_raster = f"patch_raster_{i}_{j}.tif"
        patch_path_raster = os.path.join(output_folder_raster, patch_name_raster)
        arcpy.Clip_management(resampled_tiff, patch_extent, patch_path_raster, "#", "#", "NONE")

        # 裁剪转换后的栅格
        patch_name_polygon = f"patch_polygon_{i}_{j}.tif"
        patch_path_polygon = os.path.join(output_folder_polygon, patch_name_polygon)
        arcpy.Clip_management(output_raster_result, patch_extent, patch_path_polygon, "#", "#", "NONE")

        # 设置投影信息（与输入栅格一致）
        input_projection = desc_resampled.spatialReference
        arcpy.DefineProjection_management(patch_path_raster, input_projection)
        arcpy.DefineProjection_management(patch_path_polygon, input_projection)

        print(f"裁剪完成：{patch_path_raster}, {patch_path_polygon}")

print("所有 patch 裁剪完成！")