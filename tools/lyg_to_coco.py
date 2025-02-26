# Transform Inria gt dataset (binary image) to COCO format
# Using cv2.findcontours and polygon simplify to convert raster label to vector label
#
# The first 5 images are kept as validation set

from pycocotools.coco import COCO
import os
import numpy as np
from skimage import io
import json
from tqdm import tqdm
from itertools import groupby
from shapely.geometry import Polygon, mapping
from skimage.measure import label as ski_label
from skimage.measure import regionprops
from shapely.geometry import box
import cv2
import glob
import math
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
def polygon2hbb(poly):
    """
    Get horizontal bounding box (match COCO)
    """
    p_x = poly[:, 0]
    p_y = poly[:, 1]
    hbb_x = np.min(p_x)
    hbb_y = np.min(p_y)
    hbb_w = np.around(np.max(p_x) - hbb_x, decimals=2)
    hbb_h = np.around(np.max(p_y) - hbb_y, decimals=2)
    hbox = [hbb_x, hbb_y, hbb_w, hbb_h]
    return [float(i) for i in hbox]

def clip_by_bound(poly, im_h, im_w):
    """
    Bound poly coordinates by image shape
    """
    p_x = poly[:, 0]
    p_y = poly[:, 1]
    p_x = np.clip(p_x, 0.0, im_w-1)
    p_y = np.clip(p_y, 0.0, im_h-1)
    return np.concatenate((p_x[:, np.newaxis], p_y[:, np.newaxis]), axis=1)

def crop2patch(im_p, p_h, p_w, p_overlap):
    """
    Get coordinates of upper-left point for image patch
    return: patch_list [X_upper-left, Y_upper-left, patch_width, patch_height]
    """
    im_h, im_w, _ = im_p
    x = np.arange(0, im_w-p_w, p_w-p_overlap)
    x = np.append(x, im_w-p_w)
    y = np.arange(0, im_h-p_h, p_h-p_overlap)
    y = np.append(y, im_h-p_h)
    X, Y = np.meshgrid(x, y)
    patch_list = [[i, j, p_w, p_h] for i, j in zip(X.flatten(), Y.flatten())]
    return patch_list

def polygon_in_bounding_box(polygon, bounding_box):
    """
    Returns True if all vertices of polygons are inside bounding_box
    :param polygon: [N, 2]
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    result = np.all(
        np.logical_and(
            np.logical_and(bounding_box[0] <= polygon[:, 0], polygon[:, 0] <= bounding_box[0] + bounding_box[2]),
            np.logical_and(bounding_box[1] <= polygon[:, 1], polygon[:, 1] <= bounding_box[1] + bounding_box[3])
        )
    )
    return result

def transform_poly_to_bounding_box(polygon, bounding_box):
    """
    Transform the original coordinates of polygon to bbox
    :param polygon: [N, 2]
    :param bounding_box: [row_min, col_min, row_max, col_max]
    :return:
    """
    transformed_polygon = polygon.copy()
    transformed_polygon[:, 0] -= bounding_box[0]
    transformed_polygon[:, 1] -= bounding_box[1]
    return transformed_polygon

def bmask_to_poly(b_im, simplify_ind, tolerance=1.8, ):
    """
    Convert binary mask to polygons
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    try:
        label_img = ski_label(b_im > 0)
    except:
        print('error')
    props = regionprops(label_img)
    for prop in props:
        prop_mask = np.zeros_like(b_im)
        prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
        padded_binary_mask = np.pad(prop_mask, pad_width=1, mode='constant', constant_values=0)
        contours, hierarchy = cv2.findContours(padded_binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            intp = []
            for contour, h in zip(contours, hierarchy[0]):
                contour = np.array([c.reshape(-1).tolist() for c in contour])
                # subtract pad
                contour -= 1
                contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
                if len(contour) > 3:
                    closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
                    if h[3] < 0:
                        extp = [tuple(i) for i in closed_c]
                    else:
                        if cv2.contourArea(closed_c.astype(int)) > 10:
                            intp.append([tuple(i) for i in closed_c])
            poly = Polygon(extp, intp)
            if simplify_ind:
                poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
                if isinstance(poly, Polygon):
                    polygons.append(poly)
                else:
                    for idx in range(len(poly.geoms)):
                        polygons.append(poly.geoms[idx])
        elif len(contours) == 1:
            contour = np.array([c.reshape(-1).tolist() for c in contours[0]])
            contour -= 1
            contour = clip_by_bound(contour, b_im.shape[0], b_im.shape[1])
            if len(contour) > 3:
                closed_c = np.concatenate((contour, contour[0].reshape(-1, 2)))
                poly = Polygon(closed_c)

            # simply polygon vertex
                if simplify_ind:
                    poly = poly.simplify(tolerance=tolerance, preserve_topology=False)
                if isinstance(poly, Polygon):
                    polygons.append(poly)
                else:
                    for idx in range(len(poly.geoms)):
                        polygons.append(poly.geoms[idx])
            # print(np.array(poly.exterior.coords).ravel().tolist())
            # in case that after "simplify", one polygon turn to multiply polygons
            # (pixels in polygon) are not connected
    return polygons

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def rotate_crop(im, gt, crop_size, angle):
    h, w = im.shape[0:2]
    im_rotated = rotate_image(im, angle)
    gt_rotated = rotate_image(gt, angle)
    if largest_rotated_rect(w, h, math.radians(angle))[0] > crop_size:
        im_cropped = crop_around_center(im_rotated, crop_size, crop_size)
        gt_cropped = crop_around_center(gt_rotated, crop_size, crop_size)
    else:
        print('error')
        im_cropped = crop_around_center(im, crop_size, crop_size)
        gt_cropped = crop_around_center(gt, crop_size, crop_size)
    return im_cropped, gt_cropped

def lt_crop(im, gt, crop_size):
    im_cropped = im[0:crop_size, 0:crop_size, :]
    gt_cropped = gt[0:crop_size, 0:crop_size]
    return im_cropped, gt_cropped

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


# if __name__ == '__main__':
#     input_image_path = './data/yangzhiqu/lyg_3/img/'
#     input_gt_path = './data/yangzhiqu/lyg_3/gt/'
#     save_path = './data/yangzhiqu/lyg_3/cut_512_new/'
# #     input_image_path = './data/lyg_3/cut_512/test/img/'
# #     input_gt_path = './data/lyg_3/cut_512/test/gt/'
# #     save_path = './data/lyg_3/cut_512/test'
# #     HiSup/data/lyg_3/cut_512/test

# #     cities = ['austin', 'chicago', 'kitsap', 'tyrol-w', 'vienna']
# #     val_set = [str(i) for i in range(1, 6)]

#     output_im_train = os.path.join(save_path, 'images')
#     if not os.path.exists(output_im_train):
#         os.makedirs(output_im_train)

#     patch_width = 725
#     patch_height = 725
#     patch_overlap = 200
#     patch_size = 512
#     rotation_list = [22.5, 45, 67.5]

#     # main dict for annotation file
#     output_data_train = {
#         'info': {'district': 'lyg', 'description': 'aquaculture footprints', 'contributor': 'itrslab'},
#         'categories': [{'id': 1, 'name': 'aquaculture'}],
#         'images': [],
#         'annotations': [],
#     }

#     train_ob_id = 0
#     train_im_id = 0
#     # read in data with npy format
#     input_label = os.listdir(input_gt_path)

#     for g_id, label in enumerate(tqdm(input_label)):
#         print(label)
#         # 读取数据
#         label_name = label.split('.')[0]  # 假设文件名不含扩展名以外的点
#         print(f"Processing file: {label_name}")  # 打印正在处理的文件名，以便调试
#         image_data = io.imread(os.path.join(input_image_path, label_name + '.tif'))
#         gt_im_data = io.imread(os.path.join(input_gt_path, label_name + '.tif'))
#         im_h, im_w, _ = image_data.shape

#         if True:  # 如果是训练集，开始裁切处理
#             # 获取裁切的补丁列表，使用定义好的 patch_width、patch_height 和 patch_overlap
#             patch_list = crop2patch(image_data.shape, patch_width, patch_height, patch_overlap)

#             for pid, pa in enumerate(patch_list):
#                 x_ul, y_ul, pw, ph = pa

#                 # 从 ground truth 数据中裁切出当前补丁区域
#                 p_gt = gt_im_data[y_ul:y_ul+patch_height, x_ul:x_ul+patch_width]
#                 # 从原图像数据中裁切出当前补丁区域
#                 p_im = image_data[y_ul:y_ul+patch_height, x_ul:x_ul+patch_width, :]

#                 # 创建两个列表用于存储裁切后的图像和标签
#                 p_gts = []
#                 p_ims = []

#                 # 对当前补丁进行裁切和调整，得到大小为 patch_size 的图像和标签
#                 p_im_rd, p_gt_rd = lt_crop(p_im, p_gt, patch_size)
#                 p_gts.append(p_gt_rd)
#                 p_ims.append(p_im_rd)

#                 # 对每个补丁进行旋转增强
#                 for angle in rotation_list:
#                     rot_im, rot_gt = rotate_crop(p_im, p_gt, patch_size, angle)
#                     p_gts.append(rot_gt)
#                     p_ims.append(rot_im)

#                 # 对每个图像和标签进行处理
#                 for p_im, p_gt in zip(p_ims, p_gts):
#                     # 如果标签区域中有养殖区（即标签值大于0）
#                     if np.sum(p_gt == 1) > 600:  # 这里选择大于 10 的区域作为养殖区
#                         # 将标签转为多边形
#                         p_polygons = bmask_to_poly(p_gt, 1)
#                         for poly in p_polygons:
#                             p_area = round(poly.area, 2)
#                             if p_area > 0:
#                                 # 计算边界框
#                                 p_bbox = [poly.bounds[0], poly.bounds[1],
#                                           poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]]

#                                 # 过滤掉面积太小的补丁
#                                 if p_bbox[2] > 5 and p_bbox[3] > 5:
#                                     # 获取多边形的坐标
#                                     p_seg = []
#                                     coor_list = mapping(poly)['coordinates']
#                                     for part_poly in coor_list:
#                                         p_seg.append(np.asarray(part_poly).ravel().tolist())

#                                     # 保存标签信息
#                                     anno_info = {
#                                         'id': train_ob_id,
#                                         'image_id': train_im_id,
#                                         'segmentation': p_seg,
#                                         'area': p_area,
#                                         'bbox': p_bbox,
#                                         'category_id': 1,  # 假设类别为 1
#                                         'iscrowd': 0  # 非拥挤
#                                     }
#                                     output_data_train['annotations'].append(anno_info)
#                                     train_ob_id += 1

#                     # 获取补丁信息
#                     p_name = str(train_im_id) + '.tif'

#                     # 保存补丁信息
#                     patch_info = {'id': train_im_id, 'file_name': p_name, 'width': patch_size, 'height': patch_size}
#                     output_data_train['images'].append(patch_info)

#                     # 保存补丁图像
#                     io.imsave(os.path.join(output_im_train, p_name), p_im)
#                     train_im_id += 1
                    

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     with open(os.path.join(save_path, 'annotation.json'), 'w') as f_json:
#         json.dump(output_data_train, f_json)

# ... [保留原有导入和函数定义部分]

if __name__ == '__main__':
    # 配置参数
    input_image_path = '/qiaowenjiao/HiSup/data/dongtou/geo/stretched/img'
    input_gt_path = '/qiaowenjiao/HiSup/data/dongtou/geo/stretched/gt/'
    save_path = '/qiaowenjiao/HiSup/data/dongtou/geo/cut_512_multi_yuan2/'
    output_im_train = os.path.join(save_path, 'images')
    if not os.path.exists(output_im_train):
        os.makedirs(output_im_train)

    patch_width = 725
    patch_height = 725
    patch_overlap = 200
    patch_size = 512
    rotation_list = [22.5, 45, 67.5]

    # 修改类别信息
    output_data_train = {
        'info': {'district': 'lyg', 'description': 'aquaculture footprints', 'contributor': 'itrslab'},
        'categories': [
            {'id': 1, 'name': 'raft_laver'},
            {'id': 2, 'name': 'sargassum'}
        ],
        'images': [],
        'annotations': [],
    }

    train_ob_id = 0
    train_im_id = 0
    input_label = os.listdir(input_gt_path)

#     for g_id, label in enumerate(tqdm(input_label)):
#         label_name = label.split('.')[0]
# #         image_data = io.imread(os.path.join(input_image_path, label_name + '.tif'))
#         # 修改 lyg_to_coco.py 第450行附近
#         tif_path = os.path.join(input_image_path, label_name + '.tif')
#         print(f"尝试读取文件: {tif_path}")  # 添加此行
#         image_data = io.imread(tif_path)
#         gt_im_data = io.imread(os.path.join(input_gt_path, label_name + '.tif'))

    for g_id, label in enumerate(tqdm(input_label)):
        # 安全分割文件名（处理多后缀情况）
        label_name = os.path.splitext(label)[0]  # 去掉扩展名，如 "label_001"

        try:
            # 分割文件名获取ID部分（假设文件名格式为 label_XXX.tif）
            label_parts = label_name.split('_')
            if len(label_parts) < 2:
                raise ValueError("文件名缺少下划线分隔符")

            label_id = label_parts[1]  # 提取ID部分，如 "001"

        except Exception as e:
            print(f"文件名解析失败: {label}, 错误: {e}")
            continue

        # 构建图像路径（假设图像文件命名格式为 image_XXX.tif）
        tif_filename = f"image_{label_id}.tif"  # 关键修改点
        tif_path = os.path.join(input_image_path, tif_filename)
        gt_path = os.path.join(input_gt_path, label)

        # 添加路径存在性检查
        if not os.path.exists(tif_path):
            print(f"图像文件 {tif_path} 不存在，跳过")
            continue
        if not os.path.exists(gt_path):
            print(f"标签文件 {gt_path} 不存在，跳过")
            continue

#     # 后续读取和处理代码...
#     for g_id, label in enumerate(tqdm(input_label)):
#         # 使用安全的文件名分割方式
#         label_name = os.path.splitext(label)[0]
#         label_id = label_name.splitext('_')[1]

#         # 构建图像路径
#         tif_path = os.path.join(input_image_path, 'image'+label_id + '.tif')
#         gt_path = os.path.join(input_gt_path, label)

#         # 添加路径存在性检查
#         if not os.path.exists(tif_path):
#             print(f"Error: Image file {tif_path} not found. Skipping...")
#             continue
#         if not os.path.exists(gt_path):
#             print(f"Error: GT file {gt_path} not found. Skipping...")
#             continue

        # 读取文件
        image_data = io.imread(tif_path)
        gt_im_data = io.imread(gt_path)
        im_h, im_w, _ = image_data.shape

        if True:  # 训练集处理
            patch_list = crop2patch(image_data.shape, patch_width, patch_height, patch_overlap)

            for pid, pa in enumerate(patch_list):
                x_ul, y_ul, pw, ph = pa
                p_gt = gt_im_data[y_ul:y_ul+patch_height, x_ul:x_ul+patch_width]
                p_im = image_data[y_ul:y_ul+patch_height, x_ul:x_ul+patch_width, :]

                p_gts = []
                p_ims = []
                p_im_rd, p_gt_rd = lt_crop(p_im, p_gt, patch_size)
                p_gts.append(p_gt_rd)
                p_ims.append(p_im_rd)

                for angle in rotation_list:
                    rot_im, rot_gt = rotate_crop(p_im, p_gt, patch_size, angle)
                    p_gts.append(rot_gt)
                    p_ims.append(rot_im)

                # 处理每个图像和对应的GT
                for p_im, p_gt_patch in zip(p_ims, p_gts):
                    # 遍历两个类别
                    for class_id in [1, 2]:
                        # 生成当前类别的二值掩膜
                        class_mask = (p_gt_patch == class_id).astype(np.uint8)
                        if np.sum(class_mask) > 600:
                            p_polygons = bmask_to_poly(class_mask, 1)
                            for poly in p_polygons:
                                p_area = round(poly.area, 2)
                                if p_area > 0:
                                    p_bbox = [
                                        poly.bounds[0], 
                                        poly.bounds[1],
                                        poly.bounds[2] - poly.bounds[0], 
                                        poly.bounds[3] - poly.bounds[1]
                                    ]
                                    if p_bbox[2] > 5 and p_bbox[3] > 5:
                                        coor_list = mapping(poly)['coordinates']
                                        p_seg = []
                                        for part_poly in coor_list:
                                            p_seg.append(np.asarray(part_poly).ravel().tolist())
                                        # 添加annotation，设置对应的类别ID
                                        anno_info = {
                                            'id': train_ob_id,
                                            'image_id': train_im_id,
                                            'segmentation': p_seg,
                                            'area': p_area,
                                            'bbox': p_bbox,
                                            'category_id': class_id,  # 关键修改
                                            'iscrowd': 0
                                        }
                                        output_data_train['annotations'].append(anno_info)
                                        train_ob_id += 1

                    # 保存图像信息
                    p_name = f"{train_im_id}.tif"
                    patch_info = {
                        'id': train_im_id,
                        'file_name': p_name,
                        'width': patch_size,
                        'height': patch_size
                    }
                    output_data_train['images'].append(patch_info)
                    io.imsave(os.path.join(output_im_train, p_name), p_im)
                    train_im_id += 1

    # 保存标注文件
    with open(os.path.join(save_path, 'annotation.json'), 'w') as f_json:
        json.dump(output_data_train, f_json)