# import os
# import json
# import cv2
# import numpy as np
# import argparse

# def draw_polygons_and_bboxes_on_image(image, annotations, draw_polygon=True, draw_bbox=True, line_color=(0, 255, 0),
#                                   line_thickness=1, vertex_color=(0, 0, 255), vertex_thickness=2,
#                                   bbox_color=(255, 0, 0)):
#     """
#     在图像上绘制多边形和边界框。

#     参数：
#     - image (numpy.ndarray): 输入图像。
#     - annotations (list): 包含多边形和边界框的注释列表。
#     - draw_polygon (bool): 是否绘制多边形，默认为True。
#     - draw_bbox (bool): 是否绘制边界框，默认为True。
#     - line_color (tuple): 多边形边缘的颜色（默认为绿色）。
#     - vertex_color (tuple): 顶点的颜色（默认为红色）。
#     - vertex_thickness (int): 顶点的厚度（默认为2）。
#     - bbox_color (tuple): 边界框的颜色（默认为蓝色）。

#     返回：
#     - numpy.ndarray: 绘制了多边形和边界框的图像。
#     """
#     for annotation in annotations:
#         if draw_polygon:
#             # segmentation 是一个包含多个多边形的列表，取第一个多边形进行处理
#             segmentation = annotation['segmentation'][0]
#             polygon = np.array(segmentation, np.float32).reshape((-1, 2))
#             polygon = polygon.astype(np.int32)
#             cv2.polylines(image, [polygon], isClosed=True, color=line_color, thickness=line_thickness)
#             for vertex in polygon:
#                 cv2.circle(image, tuple(vertex), vertex_thickness, vertex_color, -1)

#         if draw_bbox:
#             bbox = annotation['bbox']
#             x, y, w, h = bbox
#             top_left = (int(x), int(y))
#             bottom_right = (int(x + w), int(y + h))
#             cv2.rectangle(image, top_left, bottom_right, bbox_color, 2)

#     return image


# def process_annotations(coco_json_file, png_folder, output_folder, mask_output_folder, draw_polygon=True,
#                         draw_bbox=True, draw_mask=True):
#     """
#     处理 COCO 数据集，按需绘制多边形、边界框和掩码。

#     参数：
#     - coco_json_file (str): COCO 格式注释文件的路径。
#     - png_folder (str): 存放 PNG 图像的文件夹路径。
#     - output_folder (str): 存放带注释图像的输出文件夹路径。
#     - mask_output_folder (str): 存放掩码图像的输出文件夹路径。
#     - draw_polygon (bool): 是否在图像上绘制多边形。
#     - draw_bbox (bool): 是否在图像上绘制边界框。
#     - draw_mask (bool): 是否为图像生成掩码。
#     """
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     if not os.path.exists(mask_output_folder):
#         os.makedirs(mask_output_folder)

#     with open(coco_json_file, 'r') as f:
#         coco_data = json.load(f)

#     for image_info in coco_data['images']:
#         image_id = image_info['id']
#         file_name = image_info['file_name']
#         image_path = os.path.join(png_folder, file_name)

#         if not os.path.exists(image_path):
#             print(f"图像文件未找到: {image_path}")
#             continue

#         image = cv2.imread(image_path)
#         height, width, _ = image.shape
#         mask = np.zeros((height, width), dtype=np.uint8)
#         annotations = []

#         for annotation in coco_data['annotations']:
#             if annotation['image_id'] == image_id:
#                 annotations.append(annotation)

#         image_with_polygons_and_bboxes = draw_polygons_and_bboxes_on_image(image, annotations, draw_polygon, draw_bbox)

#         if draw_mask:
#             for annotation in annotations:
#                 segmentation = annotation['segmentation']
#                 exterior = np.array(segmentation[0]).reshape((-1, 2)).astype(np.int32)
#                 cv2.fillPoly(mask, [exterior], 255)
#                 if len(segmentation) > 1:
#                     for interior in segmentation[1:]:
#                         interior_points = np.array(interior).reshape((-1, 2)).astype(np.int32)
#                         cv2.fillPoly(mask, [interior_points], 0)

#             mask_output_path = os.path.join(mask_output_folder, file_name)
#             cv2.imwrite(mask_output_path, mask)
#             print(f"保存掩码: {mask_output_path}")

#         output_image_path = os.path.join(output_folder, file_name)
#         cv2.imwrite(output_image_path, image_with_polygons_and_bboxes)
#         print(f"保存注释图像: {output_image_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="在 COCO 数据集图像上绘制多边形、边界框和掩码。")
#     parser.add_argument("--coco_json_file", type=str, required=True, help="COCO 格式注释文件的路径。")
#     parser.add_argument("--png_folder", type=str, required=True, help="存放 PNG 图像的文件夹路径。")
#     parser.add_argument("--output_folder", type=str, required=True, help="存放带注释图像的输出文件夹路径。")
#     parser.add_argument("--mask_output_folder", type=str, required=True, help="存放掩码图像的输出文件夹路径。")
#     parser.add_argument("--draw_polygon", action="store_true", help="是否在图像上绘制多边形。")
#     parser.add_argument("--draw_bbox", action="store_true", help="是否在图像上绘制边界框。")
#     parser.add_argument("--draw_mask", action="store_true", help="是否生成图像的掩码。")
#     args = parser.parse_args()

#     process_annotations(
#         coco_json_file=args.coco_json_file,
#         png_folder=args.png_folder,
#         output_folder=args.output_folder,
#         mask_output_folder=args.mask_output_folder,
#         draw_polygon=args.draw_polygon,
#         draw_bbox=args.draw_bbox,
#         draw_mask=args.draw_mask
#     )
# # python tools/visualize_annotations.py --coco_json_file outputs/lyg_test.json \
# #                                 --png_folder data/yangzhiqu/lyg_3/cut_300/val/images \
# #                                 --output_folder outputs/images \
# #                                 --mask_output_folder outputs/masks \
# #                                 --draw_polygon --draw_bbox
import os
import cv2
import numpy as np
import argparse
import json

def draw_polygons_and_bboxes_on_image(image, predictions, draw_polygon=True, draw_bbox=True, line_color=(0, 255, 0),
                                  line_thickness=1, vertex_color=(0, 0, 255), vertex_thickness=2,
                                  bbox_color=(255, 0, 0)):
    """
    在图像上绘制多边形和边界框。

    参数：
    - image (numpy.ndarray): 输入图像。
    - predictions (list): 包含多边形和边界框的预测结果列表。
    - draw_polygon (bool): 是否绘制多边形，默认为True。
    - draw_bbox (bool): 是否绘制边界框，默认为True。
    - line_color (tuple): 多边形边缘的颜色（默认为绿色）。
    - vertex_color (tuple): 顶点的颜色（默认为红色）。
    - vertex_thickness (int): 顶点的厚度（默认为2）。
    - bbox_color (tuple): 边界框的颜色（默认为蓝色）。

    返回：
    - numpy.ndarray: 绘制了多边形和边界框的图像。
    """
    for prediction in predictions:
        if draw_polygon:
            # 处理多边形
            segmentation = prediction['segmentation'][0]  # 取第一个多边形
            polygon = np.array(segmentation, np.float32).reshape((-1, 2))
            polygon = polygon.astype(np.int32)
            cv2.polylines(image, [polygon], isClosed=True, color=line_color, thickness=line_thickness)
            for vertex in polygon:
                cv2.circle(image, tuple(vertex), vertex_thickness, vertex_color, -1)

        if draw_bbox:
            # 处理边界框
            bbox = prediction['bbox']
            x, y, w, h = bbox
            top_left = (int(x), int(y))
            bottom_right = (int(x + w), int(y + h))
            cv2.rectangle(image, top_left, bottom_right, bbox_color, 2)

    return image


def process_predictions(prediction_json_file, png_folder, output_folder, mask_output_folder, draw_polygon=True,
                        draw_bbox=True, draw_mask=True):
    """
    处理预测结果，绘制多边形、边界框和掩码。

    参数：
    - prediction_json_file (str): 预测结果文件路径，格式为json。
    - png_folder (str): 存放图片的文件夹路径。
    - output_folder (str): 存放输出图片的文件夹路径。
    - mask_output_folder (str): 存放掩码图片的文件夹路径。
    - draw_polygon (bool): 是否绘制多边形，默认为True。
    - draw_bbox (bool): 是否绘制边界框，默认为True。
    - draw_mask (bool): 是否绘制掩码，默认为True。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(mask_output_folder):
        os.makedirs(mask_output_folder)

    with open(prediction_json_file, 'r') as f:
        prediction_data = json.load(f)

    for prediction in prediction_data:
        image_id = prediction['image_id']
        file_name = f"{image_id}.png"  # 假设图片名称为 image_id.png
        image_path = os.path.join(png_folder, file_name)

        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        height, width, _ = image.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        # 绘制多边形和边界框
        image_with_polygons_and_bboxes = draw_polygons_and_bboxes_on_image(image, [prediction], draw_polygon, draw_bbox)

        if draw_mask:
            # 绘制掩码
            segmentation = prediction['segmentation']
            exterior = np.array(segmentation[0]).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [exterior], 255)

            mask_output_path = os.path.join(mask_output_folder, file_name)
            cv2.imwrite(mask_output_path, mask)
            print(f"Saved mask: {mask_output_path}")

        output_image_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_image_path, image_with_polygons_and_bboxes)
        print(f"Saved annotated image: {output_image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw polygons, bounding boxes, and masks on prediction results.")
    parser.add_argument("--prediction_json_file", type=str, required=True, help="Path to the prediction result JSON file.")
    parser.add_argument("--png_folder", type=str, required=True, help="Path to the folder containing PNG images.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save images with annotations.")
    parser.add_argument("--mask_output_folder", type=str, required=True, help="Path to save mask images.")
    parser.add_argument("--draw_polygon", action="store_true", help="Whether to draw polygons on the images.")
    parser.add_argument("--draw_bbox", action="store_true", help="Whether to draw bounding boxes on the images.")
    parser.add_argument("--draw_mask", action="store_true", help="Whether to draw masks for the images.")
    args = parser.parse_args()

    process_predictions(
        prediction_json_file=args.prediction_json_file,
        png_folder=args.png_folder,
        output_folder=args.output_folder,
        mask_output_folder=args.mask_output_folder,
        draw_polygon=args.draw_polygon,
        draw_bbox=args.draw_bbox,
        draw_mask=args.draw_mask
    )
