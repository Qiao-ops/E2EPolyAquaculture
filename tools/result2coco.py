# # import json

# # def convert_to_coco_format(results):
# #     """
# #     将检测结果转换为 COCO 格式
# #     :param results: 检测结果，格式为列表，每个元素是一个字典，包含 image_id, category_id, bbox, score 等信息
# #     :return: COCO 格式的字典
# #     """

# #     coco_result = {
# #         "images": [],
# #         "annotations": [],
# #         "categories": []
# #     }

# #     # 假设类别信息
# #     categories = [{"id": 1, "name": "category_1", "supercategory": "category"}]
# #     coco_result["categories"] = categories

# #     # 假设图像信息
# #     images_info = {
# #         4536: {
# #             "file_name": "image_4536.jpg",  # 图像文件名
# #             "width": 4167,  # 图像宽度
# #             "height": 5770  # 图像高度
# #         }
# #     }

# #     for image_id, image_info in images_info.items():
# #         coco_result["images"].append({
# #             "id": image_id,
# #             "file_name": image_info["file_name"],
# #             "width": image_info["width"],
# #             "height": image_info["height"]
# #         })

# #     # 将结果转化为 COCO 格式的 annotations
# #     annotations = []
# #     annotation_id = 1  # 用于为每个标注生成唯一 ID

# #     for result in results:
# #         image_id = result["image_id"]
# #         category_id = result["category_id"]
# #         score = result["score"]
# #         bbox = result["bbox"]  # [x, y, width, height]

# #         # 如果没有提供类别，则使用默认类别
# #         if not category_id:
# #             category_id = 1  # 默认类别 ID

# #         # 创建标注
# #         annotation = {
# #             "id": annotation_id,
# #             "image_id": image_id,
# #             "category_id": category_id,
# #             "segmentation": [],  # 如果没有分割掩码，保持为空
# #             "area": bbox[2] * bbox[3],  # 计算 area
# #             "bbox": bbox,  # [x, y, width, height]
# #             "iscrowd": 0,  # 不属于人群
# #             "score": score  # 检测分数
# #         }

# #         annotations.append(annotation)
# #         annotation_id += 1  # 递增标注 ID

# #     coco_result["annotations"] = annotations

# #     return coco_result


# # def save_to_json(coco_result, filename):
# #     """
# #     保存 COCO 格式的结果为 JSON 文件
# #     :param coco_result: COCO 格式的结果字典
# #     :param filename: 保存的文件名
# #     """
# #     with open(filename, "w") as f:
# #         json.dump(coco_result, f, indent=4)
# #     print(f"COCO 格式的结果已保存为 {filename}")


# # # 假设的检测结果：这部分可以根据你的实际情况修改
# # results = [
# #     {
# #         "image_id": 4536,
# #         "category_id": 1,
# #         "bbox": [27.078472137451172, 5.992248058319092, 60.29386901855469, 29.241987228393555],
# #         "score": 0.906668485921939
# #     },
# #     {
# #         "image_id": 4536,
# #         "category_id": 1,
# #         "bbox": [55.36922836303711, 17.641143798828125, 60.29386901855469, 29.241987228393555],
# #         "score": 0.8720561861991882
# #     }
# # ]

# # # 将检测结果转换为 COCO 格式
# # coco_result = convert_to_coco_format(results)

# # # 保存为 JSON 文件
# # save_to_json(coco_result, "output_coco_format.json")
# import json

# def convert_to_coco_format(results):
#     """
#     将检测结果转换为 COCO 格式
#     :param results: 检测结果，格式为列表，每个元素是一个字典，包含 image_id, category_id, bbox, score 等信息
#     :return: COCO 格式的字典
#     """

#     coco_result = {
#         "images": [],
#         "annotations": [],
#         "categories": []
#     }

#     # 假设类别信息
#     categories = [{"id": 1, "name": "category_1", "supercategory": "category"}]
#     coco_result["categories"] = categories

#     # 假设图像信息
#     images_info = {
#         4536: {
#             "file_name": "image_4536.jpg",  # 图像文件名
#             "width": 4167,  # 图像宽度
#             "height": 5770  # 图像高度
#         }
#     }

#     for image_id, image_info in images_info.items():
#         coco_result["images"].append({
#             "id": image_id,
#             "file_name": image_info["file_name"],
#             "width": image_info["width"],
#             "height": image_info["height"]
#         })

#     # 将结果转化为 COCO 格式的 annotations
#     annotations = []
#     annotation_id = 1  # 用于为每个标注生成唯一 ID

#     for result in results:
#         image_id = result["image_id"]
#         category_id = result["category_id"]
#         score = result["score"]
#         bbox = result["bbox"]  # [x, y, width, height]

#         # 如果没有提供类别，则使用默认类别
#         if not category_id:
#             category_id = 1  # 默认类别 ID

#         # 创建标注
#         annotation = {
#             "id": annotation_id,
#             "image_id": image_id,
#             "category_id": category_id,
#             "segmentation": [],  # 如果没有分割掩码，保持为空
#             "area": bbox[2] * bbox[3],  # 计算 area
#             "bbox": bbox,  # [x, y, width, height]
#             "iscrowd": 0,  # 不属于人群
#             "score": score  # 检测分数
#         }

#         annotations.append(annotation)
#         annotation_id += 1  # 递增标注 ID

#     coco_result["annotations"] = annotations

#     return coco_result


# def save_to_json(coco_result, filename):
#     """
#     保存 COCO 格式的结果为 JSON 文件
#     :param coco_result: COCO 格式的结果字典
#     :param filename: 保存的文件名
#     """
#     with open(filename, "w") as f:
#         json.dump(coco_result, f, indent=4)
#     print(f"COCO 格式的结果已保存为 {filename}")


# def load_results_from_json(file_path):
#     """
#     从指定路径读取检测结果的 JSON 文件
#     :param file_path: JSON 文件的路径
#     :return: 解析后的检测结果列表
#     """
#     with open(file_path, 'r') as f:
#         results = json.load(f)
#     return results


# # 从 JSON 文件加载检测结果
# results_path = "/qiaowenjiao/HiSup/outputs/lyg_hrnet48/lyg_test.json"  # 这里是存放检测结果的 JSON 文件路径
# results = load_results_from_json(results_path)

# # 将检测结果转换为 COCO 格式
# coco_result = convert_to_coco_format(results)

# # 保存为 COCO 格式的 JSON 文件
# save_to_json(coco_result, "/qiaowenjiao/HiSup/outputs/lyg_hrnet48/lyg_test_coco_format.json")
import json
import os


def convert_to_coco_format(results):
    """
    将检测结果转换为 COCO 格式
    :param results: 检测结果，格式为列表，每个元素是一个字典，包含 image_id, category_id, bbox, score 等信息
    :return: COCO 格式的字典
    """

    coco_result = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 假设类别信息
    categories = [{"id": 1, "name": "category_1", "supercategory": "category"}]
    coco_result["categories"] = categories

    # 假设图像信息（可以根据实际情况进行修改）
    images_info = {
        4536: {
            "file_name": "image_4536.jpg",  # 图像文件名
            "width": 4167,  # 图像宽度
            "height": 5770  # 图像高度
        }
    }

    for image_id, image_info in images_info.items():
        coco_result["images"].append({
            "id": image_id,
            "file_name": image_info["file_name"],
            "width": image_info["width"],
            "height": image_info["height"]
        })

    # 将结果转化为 COCO 格式的 annotations
    annotations = []
    annotation_id = 1  # 用于为每个标注生成唯一 ID

    for result in results:
        image_id = result["image_id"]
        category_id = result["category_id"]
        score = result["score"]
        bbox = result["bbox"]  # [x, y, width, height]

        # 如果没有提供类别，则使用默认类别
        if not category_id:
            category_id = 1  # 默认类别 ID

        # 创建标注
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": [],  # 如果没有分割掩码，保持为空
            "area": bbox[2] * bbox[3],  # 计算 area
            "bbox": bbox,  # [x, y, width, height]
            "iscrowd": 0,  # 不属于人群
            "score": score  # 检测分数
        }

        annotations.append(annotation)
        annotation_id += 1  # 递增标注 ID

    coco_result["annotations"] = annotations

    return coco_result


def save_to_json(coco_result, filename):
    """
    保存 COCO 格式的结果为 JSON 文件
    :param coco_result: COCO 格式的结果字典
    :param filename: 保存的文件名
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # 确保输出目录存在
    with open(filename, "w") as f:
        json.dump(coco_result, f, indent=4)
    print(f"COCO 格式的结果已保存为 {filename}")


def load_results_from_json(file_path):
    """
    从指定路径读取检测结果的 JSON 文件
    :param file_path: JSON 文件的路径
    :return: 解析后的检测结果列表
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results


def convert_json_to_coco(input_json_path, output_json_path):
    """
    从 JSON 文件加载检测结果并转换为 COCO 格式，然后保存为新的 JSON 文件
    :param input_json_path: 输入的 JSON 文件路径
    :param output_json_path: 输出的 COCO 格式 JSON 文件路径
    """
    # 从 JSON 文件加载检测结果
    results = load_results_from_json(input_json_path)

    # 将检测结果转换为 COCO 格式
    coco_result = convert_to_coco_format(results)

    # 保存为 COCO 格式的 JSON 文件
    save_to_json(coco_result, output_json_path)

if __name__ == "__main__":
    # 示例调用
    input_json_path = "/qiaowenjiao/HiSup/outputs/lyg_hrnet48/lyg_test.json"
    output_json_path = "/qiaowenjiao/HiSup/outputs/lyg_hrnet48/lyg_test_coco_format.json"

    # 将 input_json_path 中的检测结果转换为 COCO 格式并保存为 output_json_path
    convert_json_to_coco(input_json_path, output_json_path)
