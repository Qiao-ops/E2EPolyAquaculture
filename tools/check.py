import json

def check_segmentation_format(annotations):
    for ann in annotations:
        segmentation = ann.get('segmentation', [])
        
        # 检查 segmentation 是否为二维数组
        if not isinstance(segmentation, list) or not all(isinstance(s, list) for s in segmentation):
            print(f"Error: segmentation 格式不正确，image_id: {ann['image_id']}")
            continue
        
        # 检查每个坐标值是否是有效的数字，并且每个多边形的顶点数是否符合要求
        for poly in segmentation:
            if len(poly) % 2 != 0:  # 顶点数应为偶数
                print(f"Error: segmentation 顶点数不正确，image_id: {ann['image_id']}")
                continue

            if not all(isinstance(coord, (int, float)) for coord in poly):
                print(f"Error: segmentation 坐标值不正确，image_id: {ann['image_id']}")
                continue

            # 检查坐标是否在合理范围内（假设图像大小是 1000x1000）
            for i in range(0, len(poly), 2):  # 每对坐标分别检查 x 和 y
                x, y = poly[i], poly[i+1]
                if not (0 <= x <= 1000 and 0 <= y <= 1000):  # 图像的尺寸需要根据实际情况调整
                    print(f"Warning: 坐标超出图像范围，image_id: {ann['image_id']}, 坐标: ({x}, {y})")
def check_category_and_image_id_consistency(annotations, predictions):
    annotations_dict = {ann['image_id']: ann for ann in annotations}
    for pred in predictions:
        image_id = pred['image_id']
        category_id = pred['category_id']
        if image_id in annotations_dict:
            ann = annotations_dict[image_id]
            if ann['category_id'] != category_id:
                print(f"警告: 预测结果中的 category_id ({category_id}) 与标签中的 category_id ({ann['category_id']}) 不匹配，image_id: {image_id}")
                
if __name__ == "__main__":




#         # 加载标签文件和预测结果文件
#     with open('/qiaowenjiao/HiSup/data/yangzhiqu/lyg_3/cut_300/val/annotation.json', 'r') as f:
#         annotations_data = json.load(f)
#     # 加载预测结果文件
#     with open('/qiaowenjiao/HiSup/outputs/lyg_hrnet48/lyg_test.json', 'r') as f:
#         predictions = json.load(f)
# import json

    # 定义 JSON 文件路径
    annotations_file = '/qiaowenjiao/HiSup/data/yangzhiqu/lyg_3/cut_300/val/annotation.json'
    predictions_file = '/qiaowenjiao/HiSup/outputs/lyg_hrnet48/lyg_test.json'

    # 尝试加载标签文件并捕获异常
    try:
        with open(annotations_file, 'r') as f:
            annotations_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {annotations_file}")
        exit(1)
    except json.JSONDecodeError:
        print(f"错误：无法解析 JSON 文件 {annotations_file}")
        exit(1)

    # 尝试加载预测文件并捕获异常
    try:
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {predictions_file}")
        exit(1)
    except json.JSONDecodeError:
        print(f"错误：无法解析 JSON 文件 {predictions_file}")
        exit(1)

    # 打印annotations_data的类型和键
    print(f"annotations_data类型: {type(annotations_data)}")
    print(f"annotations_data的键: {annotations_data.keys()}")  # 打印字典的键

    # 获取 annotations 部分
    annotations = annotations_data.get('annotations', [])
    # 获取预测的图像 ID
    predictions = predictions_data
    prediction_image_ids = {pred['image_id'] for pred in predictions}
    print("prediction_image_ids",prediction_image_ids)

    # 获取标签中的 image_id
    annotation_image_ids = {ann['image_id'] for ann in annotations}

    # 检查 predictions 中的 image_id 是否存在于 annotations 中
    missing_image_ids = prediction_image_ids - annotation_image_ids
    if missing_image_ids:
        print(f"以下 image_id 在标签文件中不存在: {missing_image_ids}")
    else:
        print("所有预测的 image_id 在标签文件中都有对应项")

# 继续检查 segmentation 和 bbox 格式是否一致

#     # 检查 predictions 的类型
#     print(type(predictions))
#     print(predictions[:5])  # 打印前 5 个元素，查看结构
#     with open('/qiaowenjiao/HiSup/data/yangzhiqu/lyg_3/cut_300/val/annotation.json', 'r') as f:
#         predictions = json.load(f)

#     # 获取标签文件和预测结果的 image_id 列表
#     annotation_image_ids = {ann['image_id'] for ann in annotations}
#     prediction_image_ids = {pred['image_id'] for pred in predictions}

#     # 检查 image_id 是否一致
#     missing_in_annotations = prediction_image_ids - annotation_image_ids
#     missing_in_predictions = annotation_image_ids - prediction_image_ids

#     if missing_in_annotations:
#         print(f"预测结果中缺失的 image_id: {missing_in_annotations}")
#     else:
#         print("所有预测的 image_id 都在标签中找到。")

#     if missing_in_predictions:
#         print(f"标签中缺失的 image_id: {missing_in_predictions}")
#     else:
#         print("所有标签的 image_id 都在预测结果中找到。")
#     # 获取标签文件和预测结果的 image_id 列表
#     annotation_image_ids = {ann['image_id'] for ann in annotations}
#     prediction_image_ids = {pred['image_id'] for pred in predictions}

#     # 检查 image_id 是否一致
#     missing_in_annotations = prediction_image_ids - annotation_image_ids
#     missing_in_predictions = annotation_image_ids - prediction_image_ids

#     if missing_in_annotations:
#         print(f"预测结果中缺失的 image_id: {missing_in_annotations}")
#     else:
#         print("所有预测的 image_id 都在标签中找到。")

#     if missing_in_predictions:
#         print(f"标签中缺失的 image_id: {missing_in_predictions}")
#     else:
#         print("所有标签的 image_id 都在预测结果中找到。")
#     # 检查标签文件和预测结果中的 category_id 一致性
#     # 确保 predictions 是列表类型
#     if isinstance(predictions, list):
#         print("Predictions 格式正确")
#     else:
#         print("Predictions 格式错误")

#     # 获取标签中的 image_id 列表
#     annotation_image_ids = {ann['image_id'] for ann in annotations}

#     # 获取预测结果中的 image_id 列表
#     prediction_image_ids = {pred['image_id'] for pred in predictions}

#     # 检查 image_id 是否一致
#     missing_in_annotations = prediction_image_ids - annotation_image_ids
#     missing_in_predictions = annotation_image_ids - prediction_image_ids

#     if missing_in_annotations:
#         print(f"预测结果中缺失的 image_id: {missing_in_annotations}")
#     else:
#         print("所有预测的 image_id 都在标签中找到。")

#     if missing_in_predictions:
#         print(f"标签中缺失的 image_id: {missing_in_predictions}")
#     else:
#         print("所有标签的 image_id 都在预测结果中找到。")

    # 加载标签文件
#     with open('path_to_annotations.json', 'r') as f:
#         annotations = json.load(f)

#     # 检查annotations的类型和内容
#     print(f"annotations类型: {type(annotations)}")
#     print(f"annotations前5项: {annotations[:5]}")  # 打印前5项，检查格式
#     # 检查annotations的类型和内容
#     print(f"annotations类型: {type(annotations)}")
#     print(f"annotations的键: {annotations.keys()}")  # 打印字典的键，看看有哪些

#     # 如果 annotations 是字典，尝试访问它的键
#     if isinstance(annotations, dict):
#         print("annotations是字典，检查它的内容：")
#         for key, value in annotations.items():
#             print(f"键: {key}, 值类型: {type(value)}")
#             if isinstance(value, list):
#                 print(f"列表的长度: {len(value)}")
#                 print(f"列表的前5项: {value[:5]}")
#                 break  # 只打印第一个列表，避免输出过多

#     # 确保 annotations 是一个列表
#     if isinstance(annotations, list):
#         print("annotations 格式正确")
#     else:
#         print("annotations 格式错误")

#     # 获取标签中的 image_id 列表
#     annotation_image_ids = {ann['image_id'] for ann in annotations}

#     # 获取预测结果中的 image_id 列表
#     prediction_image_ids = {pred['image_id'] for pred in predictions}

#     # 检查 image_id 是否一致
#     missing_in_annotations = prediction_image_ids - annotation_image_ids
#     missing_in_predictions = annotation_image_ids - prediction_image_ids

#     if missing_in_annotations:
#         print(f"预测结果中缺失的 image_id: {missing_in_annotations}")
#     else:
#         print("所有预测的 image_id 都在标签中找到。")

#     if missing_in_predictions:
#         print(f"标签中缺失的 image_id: {missing_in_predictions}")
#     else:
#         print("所有标签的 image_id 都在预测结果中找到。")
# # 打印annotations_data的结构
#     print(f"annotations_data类型: {type(annotations_data)}")
#     print(f"annotations_data的键: {annotations_data.keys()}")  # 打印字典的键

#     # 获取annotations部分
#     annotations = annotations_data.get('annotations', [])

#     # 打印annotations的前5项以检查格式
#     print(f"annotations前5项: {annotations[:5]}")

#     # 获取 image_ids
#     annotation_image_ids = {ann['image_id'] for ann in annotations}
#     print(f"annotation_image_ids: {annotation_image_ids}")

#     check_category_and_image_id_consistency(annotations, predictions)

#     # 检查标签文件中的 segmentation 格式
#     check_segmentation_format(annotations)
