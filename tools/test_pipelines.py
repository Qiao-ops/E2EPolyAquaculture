

import os
import os.path as osp
import json
import torch
import logging
import numpy as np
import scipy
import scipy.ndimage

from PIL import Image
from tqdm import tqdm
from skimage import io
from tools.evaluation import coco_eval, boundary_eval, polis_eval
from hisup.utils.comm import to_single_device
from hisup.utils.polygon import generate_polygon_new
from hisup.utils.visualizer import *
from hisup.dataset import build_test_dataset
from hisup.dataset.build import build_transform
from hisup.utils.polygon import juncs_in_bbox

from shapely.geometry import Polygon
from skimage.measure import label, regionprops

from pycocotools import mask as coco_mask
import tifffile
import re


def poly_to_bbox(poly):
    """
    input: poly----2D array with points
    """
    lt_x = np.min(poly[:,0])
    lt_y = np.min(poly[:,1])
    w = np.max(poly[:,0]) - lt_x
    h = np.max(poly[:,1]) - lt_y
    return [float(lt_x), float(lt_y), float(w), float(h)]

def generate_coco_ann(polys, scores, img_id):
    sample_ann = []
    for i, polygon in enumerate(polys):
        if polygon.shape[0] < 3:
            continue

        vec_poly = polygon.ravel().tolist()
        poly_bbox = poly_to_bbox(polygon)
        ann_per_building = {
                'image_id': img_id,
                'category_id': 100,
                'segmentation': [vec_poly],
                'bbox': poly_bbox,
                'score': float(scores[i]),
            }
        sample_ann.append(ann_per_building)

    return sample_ann

def generate_coco_mask(mask, img_id):
    sample_ann = []
    props = regionprops(label(mask > 0.50))
    for prop in props:
        if ((prop.bbox[2] - prop.bbox[0]) > 0) & ((prop.bbox[3] - prop.bbox[1]) > 0):
            prop_mask = np.zeros_like(mask, dtype=np.uint8)
            prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1

            masked_instance = np.ma.masked_array(mask, mask=(prop_mask != 1))
            score = masked_instance.mean()
            encoded_region = coco_mask.encode(np.asfortranarray(prop_mask))
            ann_per_building = {
                'image_id': img_id,
                'category_id': 100,
                'segmentation': {
                    "size": encoded_region["size"],
                    "counts": encoded_region["counts"].decode()
                },
                'score': float(score),
            }
            sample_ann.append(ann_per_building)

    return sample_ann
def truncate_stretch(image, min_percentile=2, max_percentile=98):
    stretched_channels = []
    # 排除所有通道全为0的像素
    image_n0 = image[np.any(image != 0, axis=-1)]
    for channel in range(image.shape[2]):
        image_channel = image[:, :, channel]

        # 计算每个通道的百分位数
        min_val = np.percentile(image_n0[:, channel], min_percentile)
        max_val = np.percentile(image_n0[:, channel], max_percentile)
        print("min: ", min_val)
        print("max: ", max_val)
        # 将像素值限制在截断范围内
        clipped_channel = np.clip(image_channel, min_val, max_val)

        # 缩放到0-255范围内
        stretched_channel = ((clipped_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        stretched_channels.append(stretched_channel)

    # 将各个通道重新堆叠成图像
    stretched_image = np.stack(stretched_channels, axis=2)

    return stretched_image

class TestPipeline():
    def __init__(self, cfg, eval_type='coco_iou'):
        self.cfg = cfg
        self.device = cfg.MODEL.DEVICE
        self.output_dir = cfg.OUTPUT_DIR
        self.dataset_name = cfg.DATASETS.TEST[0]
        self.eval_type = eval_type
        
        self.gt_file = ''
        self.dt_file = ''
    
    def test(self, model):
        if 'crowdai' in self.dataset_name:
            self.test_on_crowdai(model, self.dataset_name)
        elif 'inria' in self.dataset_name:
            self.test_on_inria(model, self.dataset_name)
#             self.test_on_crowdai(model, self.dataset_name)

        else:
            self.test_on_inria(model, self.dataset_name)
#             self.test_on_crowdai(model, self.dataset_name)

    def eval(self):
        logger = logging.getLogger("testing")
        logger.info('Evalutating on {}'.format(self.eval_type))
        if self.eval_type == 'coco_iou':
            coco_eval(self.gt_file, self.dt_file)
        elif self.eval_type == 'boundary_iou':
            boundary_eval(self.gt_file, self.dt_file)
        elif self.eval_type == 'polis':
            polis_eval(self.gt_file, self.dt_file)

    def test_on_crowdai(self, model, dataset_name):
        logger = logging.getLogger("testing")
        logger.info('Testing on {} dataset'.format(dataset_name))
        
        results = []
        mask_results = []
        test_dataset, gt_file = build_test_dataset(self.cfg)
        for i, (images, annotations) in enumerate(tqdm(test_dataset)):
            with torch.no_grad():
                output, _ = model(images.to(self.device), to_single_device(annotations, self.device))
                output = to_single_device(output,'cpu')

            batch_size = images.size(0)
            batch_scores = output['scores']
            batch_polygons = output['polys_pred']
            batch_masks = output['mask_pred']

            for b in range(batch_size):
                filename = annotations[b]['filename']
#                 img_id = int(filename[:-4])
#                 img_id = int(filename.split('.')[0])  # 假设文件名不含扩展名以外的点
                match = re.search(r'(\d+)', filename)
                if match:
                    img_id = int(match.group(1))  # 获取匹配到的数字部分
#                     print(img_id.type)
#                 scores = batch_scores[b]
                print(f"Batch size: {len(batch_scores)}")
                print(f"Index b: {b}")
#                 if b < len(batch_scores):
#                     scores = batch_scores[b]
#                 else:
#                     print(f"Invalid index {b} for batch_scores of size {len(batch_scores)}")
#                     # 处理索引越界的情况，例如跳过当前批次，或者退出
                if len(batch_scores) > 0 and b < len(batch_scores):
                    scores = batch_scores[b]
                else:
                    print(f"Skipping empty batch or invalid index {b}")
                    continue  # 跳过当前空批次，继续处理下一个

                polys = batch_polygons[b]
                mask_pred = batch_masks[b]

                image_result = generate_coco_ann(polys, scores, img_id)
                if len(image_result) != 0:
                    results.extend(image_result)

                image_masks = generate_coco_mask(mask_pred, img_id)
                if len(image_masks) != 0:
                    mask_results.extend(image_masks)
        
        dt_file = osp.join(self.output_dir,'{}.json'.format(dataset_name))
        logger.info('Writing the results of the {} dataset into {}'.format(dataset_name,
                    dt_file))
        with open(dt_file,'w') as _out:
            json.dump(results,_out)

        self.gt_file = gt_file
        self.dt_file = dt_file
        self.eval()

        dt_file = osp.join(self.output_dir,'{}_mask.json'.format(dataset_name))
        logger.info('Writing the results of the {} dataset into {}'.format(dataset_name,
                    dt_file))
        with open(dt_file,'w') as _out:
            json.dump(mask_results,_out)

        self.gt_file = gt_file
        self.dt_file = dt_file
        self.eval()

    def test_on_inria(self, model, dataset_name):
        logger = logging.getLogger("testing")
        logger.info('Testing on {} dataset'.format(dataset_name))

#         IM_PATH = './data/yangzhiqu/lyg_3/test'
        
        IM_PATH = '/qiaowenjiao/HiSup/data/yangzhiqu/lyg_3/val8'
#         IM_PATH = './data/yangzhiqu/lyg_3/cut_512_new/images'
#         IM_PATH = '/qiaowenjiao/HiSup/data/lyg_3/geo/cut_2048_u8/img'
        if not os.path.exists(os.path.join(self.output_dir, 'seg')):
            os.makedirs(os.path.join(self.output_dir, 'seg'))
        transform = build_transform(self.cfg)
        test_imgs = os.listdir(IM_PATH)
        # 过滤非图像文件
        test_imgs = [f for f in test_imgs if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))]
        for image_name in tqdm(test_imgs, desc='Total processing'):
            file_name = image_name
            
            impath = osp.join(IM_PATH, file_name)
            print(f"Processing: {impath}")  # 打印 impath 以验证

            image = io.imread(impath)
#             image = io.imread(impath, plugin='tiff')
#             image = tifffile.imread(impath)

            # crop the original inria image(5000x5000) into small images(512x512)
#             h_stride, w_stride = 400, 400
            h_stride, w_stride = 200, 200

#             h_crop, w_crop = 512, 512
            h_crop, w_crop = 300, 300

            h_img, w_img, _ = image.shape
            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
            pred_whole_img = np.zeros([h_img, w_img], dtype=np.float32)
            count_mat = np.zeros([h_img, w_img])
            # weight = np.zeros([h_img, w_img])
            juncs_whole_img = []
            
            patch_weight = np.ones((h_crop + 2, w_crop + 2))
            patch_weight[0,:] = 0
            patch_weight[-1,:] = 0
            patch_weight[:,0] = 0
            patch_weight[:,-1] = 0
            
            patch_weight = scipy.ndimage.distance_transform_edt(patch_weight)
            patch_weight = patch_weight[1:-1,1:-1]

            for h_idx in tqdm(range(h_grids), leave=False, desc='processing on per image'):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)
                    
                    crop_img = image[y1:y2, x1:x2, :]
                    crop_img_tensor = transform(crop_img.astype(float))[None].to(self.device)
                    
                    meta = {
                        'filename': impath,
                        'height': crop_img.shape[0],
                        'width': crop_img.shape[1],
                        'pos': [x1, y1, x2, y2]
                    }

                    with torch.no_grad():
                        output, _ = model(crop_img_tensor, [meta])
                        output = to_single_device(output, 'cpu')

                    juncs_pred = output['juncs_pred'][0]
                    juncs_pred += [x1, y1]
                    juncs_whole_img.extend(juncs_pred.tolist())
                    mask_pred = output['mask_pred'][0]
                    mask_pred *= patch_weight
                    pred_whole_img += np.pad(mask_pred,
                                        ((int(y1), int(pred_whole_img.shape[0] - y2)),
                                        (int(x1), int(pred_whole_img.shape[1] - x2))))
                    count_mat[y1:y2, x1:x2] += patch_weight

            juncs_whole_img = np.array(juncs_whole_img)
            pred_whole_img = pred_whole_img / count_mat

            # match junction and seg results
            polygons = []
            props = regionprops(label(pred_whole_img > 0.5))
            for prop in tqdm(props, leave=False, desc='polygon generation'):
                y1, x1, y2, x2 = prop.bbox
                bbox = [x1, y1, x2, y2]
                select_juncs = juncs_in_bbox(bbox, juncs_whole_img, expand=8)
                poly, juncs_sa, _, _, juncs_index = generate_polygon_new(prop, pred_whole_img, \
                                                                          select_juncs, pid=0, test_inria=True)
                if juncs_sa.shape[0] == 0:
                    continue
                
                if len(juncs_index) == 1:
                    polygons.append(Polygon(poly))
                else:
                    poly_ = Polygon(poly[juncs_index[0]], \
                                    [poly[idx] for idx in juncs_index[1:]])
                    polygons.append(poly_)

            # visualize
            # viz_inria(image, polygons, self.cfg.OUTPUT_DIR, file_name)
#             save_viz(image, polygons, self.cfg.OUTPUT_DIR, file_name)
            save_dir = self.cfg.OUTPUT_DIR
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
#             save_viz(image, polygons,save_dir, file_name)
#             save_shapefile_from_polygons(polygons, image_path, save_dir, resolution= 1.8788188)
            # 假设你已经生成了多边形列表 polygons

            # 输入图像的路径
            image_path = impath  # 这里使用 impath，因为它是当前处理的图像路径

            # 输出 Shapefile 的路径
            output_shapefile_path = os.path.join(save_dir,'shp',f"{file_name.split('.')[0]}_polygons.shp")

            # 分辨率（根据你的图像分辨率设置）
            resolution= 1.8788188
            # 调用函数保存 Shapefile
            save_shapefile_from_polygons(polygons, image_path, output_shapefile_path, resolution)

            # 可视化并保存结果
            save_viz(image, polygons, save_dir, file_name)
            # save seg results
            #im = Image.fromarray(pred_whole_img)
            im = Image.fromarray(((pred_whole_img >0.5) * 255).astype(np.uint8), 'L')
            im.save(os.path.join(self.output_dir, 'seg', file_name))
            