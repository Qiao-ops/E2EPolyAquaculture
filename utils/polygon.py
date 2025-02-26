import cv2
import torch
import numpy as np
import torch.nn.functional as F

from scipy.spatial.distance import cdist

def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

def get_junctions(jloc, joff, topk = 300, th=0):
    height, width = jloc.size(1), jloc.size(2)
    jloc = jloc.reshape(-1)
    joff = joff.reshape(2, -1)

    scores, index = torch.topk(jloc, k=topk)
    y = (index // width).float() + torch.gather(joff[1], 0, index) + 0.5
    x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5

    junctions = torch.stack((x, y)).t()

    return junctions[scores>th], scores[scores>th]

def get_pred_junctions(convex_map, concave_map, joff_map):
    # concave junctions
    concave_pred_nms = non_maximum_suppression(concave_map)
    topK = min(300, int((concave_pred_nms>0.008).float().sum().item()))
    juncs_concave, _ = get_junctions(concave_pred_nms, joff_map, topk=topK)
    # convex junctions
    convex_pred_nms = non_maximum_suppression(convex_map)
    topK = min(300, int((convex_pred_nms > 0.008).float().sum().item()))
    juncs_convex, _ = get_junctions(convex_pred_nms, joff_map, topk=topK)

    juncs_pred = torch.cat((juncs_concave, juncs_convex), 0)

    return juncs_pred.detach().cpu().numpy()

def juncs_in_bbox(bbox, juncs, expand=8):
    bbox = np.array([bbox[0]-expand, bbox[1]-expand, bbox[2]+expand, bbox[3]+expand]).clip(0, 5000)
    ll = np.array([bbox[0],bbox[1]])
    ur = np.array([bbox[2],bbox[3]])
    index = np.all((ll <= juncs) & (juncs <= ur), axis=1)
    return juncs[index]

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def ext_c_to_poly_coco(ext_c, im_h, im_w):
    mask = np.zeros([im_h+1, im_w+1], dtype=np.uint8)
    polygon = np.int0(ext_c)
    cv2.drawContours(mask, [polygon.reshape(-1, 1, 2)], -1, color=1, thickness=-1)
    trans_prop_mask = mask.copy()
    f_y, f_x = np.where(mask == 1)
    trans_prop_mask[f_y + 1, f_x] = 1
    trans_prop_mask[f_y, f_x + 1] = 1
    trans_prop_mask[f_y + 1, f_x + 1] = 1
    contours, _ = cv2.findContours(trans_prop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)
    poly = np.concatenate((contour, contour[0].reshape(-1, 2)))
    new_poly = diagonal_to_square(poly)
    return new_poly

def diagonal_to_square(poly):
    new_c = []
    for id, p in enumerate(poly[:-1]):
        if (p[0] + 1 == poly[id + 1][0] and p[1] == poly[id + 1][1]) \
                or (p[0] == poly[id + 1][0] and p[1] + 1 == poly[id + 1][1]) \
                or (p[0] - 1 == poly[id + 1][0] and p[1] == poly[id + 1][1]) \
                or (p[0] == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
        elif (p[0] + 1 == poly[id + 1][0] and p[1] + 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0] + 1, p[1]])
        elif (p[0] - 1 == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0] - 1, p[1]])
        elif (p[0] + 1 == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0], p[1] - 1])
        else:
            new_c.append(p)
            new_c.append([p[0], p[1] + 1])
    new_poly = np.asarray(new_c)
    new_poly = np.concatenate((new_poly, new_poly[0].reshape(-1, 2)))
    return new_poly

def inn_c_to_poly_coco(inn_c, im_h, im_w):
    mask = np.zeros([im_h + 1, im_w + 1], dtype=np.uint8)
    polygon = np.int0(inn_c)
    cv2.drawContours(mask, [polygon.reshape(-1, 1, 2)], -1, color=1, thickness=-1)
    trans_prop_mask = mask.copy()
    f_y, f_x = np.where(mask == 1)
    trans_prop_mask[f_y[np.where(f_y == min(f_y))], f_x[np.where(f_y == min(f_y))]] = 0
    trans_prop_mask[f_y[np.where(f_x == min(f_x))], f_x[np.where(f_x == min(f_x))]] = 0
    #trans_prop_mask[max(f_y), max(f_x)] = 1
    contours, _ = cv2.findContours(trans_prop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)[::-1]
    poly = np.concatenate((contour, contour[0].reshape(-1, 2)))
    #return poly
    new_poly = diagonal_to_square(poly)
    return new_poly

def simple_polygon(poly, thres=20):
    if (poly[0] == poly[-1]).all():
        poly = poly[:-1]
    lines = np.concatenate((poly, np.roll(poly, -1, axis=0)), axis=1)
    vec0 = lines[:, 2:] - lines[:, :2]
    vec1 = np.roll(vec0, -1, axis=0)
    vec0_ang = np.arctan2(vec0[:,1], vec0[:,0]) * 180 / np.pi
    vec1_ang = np.arctan2(vec1[:,1], vec1[:,0]) * 180 / np.pi
    lines_ang = np.abs(vec0_ang - vec1_ang)

    flag1 = np.roll((lines_ang > thres), 1, axis=0)
    flag2 = np.roll((lines_ang < 360 - thres), 1, axis=0)
    simple_poly = poly[np.bitwise_and(flag1, flag2)]
    simple_poly = np.concatenate((simple_poly, simple_poly[0].reshape(-1,2)))
    return simple_poly
def simple_polygon_new(poly, thres=10, epsilon=1.0):
    """
    简化多边形，结合角度阈值和 Douglas-Peucker 算法。
    
    参数:
        poly (np.ndarray): 输入多边形的顶点坐标，形状为 (N, 2)。
        thres (float): 角度阈值，用于过滤冗余点。
        epsilon (float): Douglas-Peucker 算法的简化阈值。
    
    返回:
        np.ndarray: 简化后的多边形顶点坐标。
    """
    # 检查输入多边形的有效性
    if not isinstance(poly, np.ndarray) or poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("输入多边形必须是形状为 (N, 2) 的 NumPy 数组。")
    if len(poly) < 3:
        return poly  # 点数少于 3，无法形成多边形，直接返回

    # 如果多边形是闭合的，去掉最后一个重复点
    if (poly[0] == poly[-1]).all():
        poly = poly[:-1]

    # 使用角度阈值简化多边形
    simple_poly = angle_based_simplification(poly, thres)

    # 如果简化后的多边形点数少于 3，直接返回原多边形
    if len(simple_poly) < 3:
        simple_poly = poly

    # 使用 Douglas-Peucker 算法进一步简化
    if len(simple_poly) > 2:
        simple_poly = douglas_peucker_simplification(simple_poly, epsilon)

    # 确保多边形是闭合的
    if not (simple_poly[0] == simple_poly[-1]).all():
        simple_poly = np.concatenate((simple_poly, simple_poly[0].reshape(-1, 2)))

    return simple_poly

def angle_based_simplification(poly, thres):
    """
    使用角度阈值简化多边形。
    
    参数:
        poly (np.ndarray): 输入多边形的顶点坐标，形状为 (N, 2)。
        thres (float): 角度阈值。
    
    返回:
        np.ndarray: 简化后的多边形顶点坐标。
    """
    # 将多边形的顶点与其下一个顶点连接成线段
    lines = np.concatenate((poly, np.roll(poly, -1, axis=0)), axis=1)
    
    # 计算每条线段的向量
    vec0 = lines[:, 2:] - lines[:, :2]
    
    # 计算下一条线段的向量
    vec1 = np.roll(vec0, -1, axis=0)
    
    # 计算每条线段的角度（以度为单位）
    vec0_ang = np.arctan2(vec0[:, 1], vec0[:, 0]) * 180 / np.pi
    vec1_ang = np.arctan2(vec1[:, 1], vec1[:, 0]) * 180 / np.pi
    
    # 计算相邻线段之间的角度差
    lines_ang = np.abs(vec0_ang - vec1_ang)
    
    # 处理角度差大于 180 度的情况
    lines_ang = np.where(lines_ang > 180, 360 - lines_ang, lines_ang)
    
    # 标记需要保留的顶点
    flag1 = np.roll((lines_ang > thres), 1, axis=0)  # 当前顶点与下一个顶点的角度差大于阈值
    flag2 = np.roll((lines_ang < 360 - thres), 1, axis=0)  # 当前顶点与下一个顶点的角度差小于 360 - 阈值
    
    # 保留满足条件的顶点
    simple_poly = poly[np.bitwise_and(flag1, flag2)]
    
    # 确保多边形是闭合的
    if not (simple_poly[0] == simple_poly[-1]).all():
        simple_poly = np.concatenate((simple_poly, simple_poly[0].reshape(-1, 2)))
    
    return simple_poly

def douglas_peucker_simplification(poly, epsilon):
    """
    使用 Douglas-Peucker 算法简化多边形。
    
    参数:
        poly (np.ndarray): 输入多边形的顶点坐标，形状为 (N, 2)。
        epsilon (float): 简化阈值，控制简化程度。
    
    返回:
        np.ndarray: 简化后的多边形顶点坐标。
    """
    # 将多边形转换为闭合形式
    if not (poly[0] == poly[-1]).all():
        poly = np.concatenate((poly, poly[0].reshape(-1, 2)))
    
    # 使用 OpenCV 的 approxPolyDP 进行简化
    simplified_poly = cv2.approxPolyDP(poly.astype(np.float32), epsilon, closed=True)
    
    # 去掉多余的维度并返回
    return simplified_poly.squeeze()
def generate_polygon_new(prop, mask, juncs, pid, test_inria):
    # 获取图像块的分辨率
    im_h, im_w = mask.shape

    # 动态调整角度阈值和 epsilon 值
    thres = max(5, 10 - (im_h * im_w) / 1e8)  # 分母从 1e6 调整为 1e8
    epsilon = max(0.5, 1.0 + (im_h * im_w) / 1e8)  # 分母从 1e6 调整为 1e8

    if not test_inria:
        poly, score, juncs_pred_index = get_poly_crowdai(prop, mask, juncs)
        # 调用 simple_polygon 对多边形进行简化
        poly = simple_polygon_new(poly, thres=thres, epsilon=epsilon)
        juncs_sa, edges_sa, junc_index = get_junc_edge_id(poly, juncs_pred_index)
        return poly, juncs_sa, edges_sa, score, juncs_pred_index
    else:
        poly, score, juncs_pred_index = get_poly_inria(prop, mask, juncs, pid)
        # 调用 simple_polygon 对多边形进行简化
        poly = simple_polygon_new(poly, thres=thres, epsilon=epsilon)
        edges_sa = get_edge_id_inria(poly, juncs_pred_index) - pid
        return poly, poly, edges_sa, score, juncs_pred_index

def generate_polygon(prop, mask, juncs, pid, test_inria):
    if not test_inria:
        poly, score, juncs_pred_index = get_poly_crowdai(prop, mask, juncs)
        juncs_sa, edges_sa, junc_index = get_junc_edge_id(poly, juncs_pred_index)
        return poly, juncs_sa, edges_sa, score, juncs_pred_index
    else:
        poly, score, juncs_pred_index = get_poly_inria(prop, mask, juncs, pid)
        edges_sa = get_edge_id_inria(poly, juncs_pred_index) - pid
        return poly, poly, edges_sa, score, juncs_pred_index


def get_poly_crowdai(prop, mask_pred, junctions):
    prop_mask = np.zeros_like(mask_pred).astype(np.uint8)
    prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
    masked_instance = np.ma.masked_array(mask_pred, mask=(prop_mask != 1))
    score = masked_instance.mean()
    im_h, im_w = mask_pred.shape
    contours, hierarchy = cv2.findContours(prop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    poly = []
    edge_index = []
    jid = 0
    for contour, h in zip(contours, hierarchy[0]):
        c = []
        if h[3] == -1:
            c = ext_c_to_poly_coco(contour, im_h, im_w)
        if h[3] != -1:
            if cv2.contourArea(contour) >= 50:
                c = inn_c_to_poly_coco(contour, im_h, im_w)
            #c = inn_c_to_poly_coco(contour, im_h, im_w)
        if len(c) > 3:
            init_poly = c.copy()
            if len(junctions) > 0:
                cj_match_ = np.argmin(cdist(c, junctions), axis=1)
                cj_dis = cdist(c, junctions)[np.arange(len(cj_match_)), cj_match_]
                u, ind = np.unique(cj_match_[cj_dis < 5], return_index=True)
                if len(u) > 2:
                    ppoly = junctions[u[np.argsort(ind)]]
                    ppoly = np.concatenate((ppoly, ppoly[0].reshape(-1, 2)))
                    init_poly = ppoly
            init_poly = simple_polygon(init_poly, thres=10)
            poly.extend(init_poly.tolist())
            edge_index.append([i for i in range(jid, jid+len(init_poly)-1)])
            jid += len(init_poly)
    return np.array(poly), score, edge_index

def get_poly_inria(prop, mask_pred, junctions, pid):
    prop_mask = np.zeros_like(mask_pred).astype(np.uint8)
    prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1
    masked_instance = np.ma.masked_array(mask_pred, mask=(prop_mask != 1))
    score = masked_instance.mean()
    contours, hierarchy = cv2.findContours(prop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    poly_junc = []
    junc_index = []
    jid = pid
    for contour, h in zip(contours, hierarchy[0]):
        contour = np.array([c.reshape(-1).tolist() for c in contour])
        if len(contour) > 2:
            init_poly = contour.copy()
            if len(junctions) > 0:
                cj_match_ = np.argmin(cdist(contour, junctions), axis=1)
                cj_dis = cdist(contour, junctions)[np.arange(len(cj_match_)), cj_match_]
                u, ind = np.unique(cj_match_[cj_dis < 5], return_index=True)
                if len(u) > 2:
                    ppoly = junctions[u[np.argsort(ind)]]
                    #ppoly = np.concatenate((ppoly, ppoly[0].reshape(-1, 2)))
                    init_poly = ppoly
            init_poly = simple_polygon(init_poly, thres=20)
            poly_junc.extend(init_poly)
            junc_index.append(np.array([i for i in range(jid, jid+len(init_poly))]))
            jid += len(init_poly)
    return np.array(poly_junc), score, junc_index

def get_junc_edge_id(poly, junc_pred_index):
    juncs_proj = []
    junc_index = []
    edges_sa = []
    jid = 0
    for ci, cc in enumerate(junc_pred_index):
        juncs_proj.extend(poly[cc])
        if ci == 0:
            junc_id = np.asarray(cc).reshape(-1, 1)
            jid += len(junc_id)
        else:
            junc_id = np.arange(jid, jid+len(cc)).reshape(-1, 1)
            jid += len(cc)
        edge_id = np.concatenate((junc_id, np.roll(junc_id, -1)), axis=1)
        edges_sa.extend(edge_id)
        junc_index.append(junc_id.ravel())
    edges_sa = np.asarray(edges_sa)
    juncs_proj = np.asarray(juncs_proj)
    return juncs_proj, edges_sa, junc_index

def get_edge_id_inria(poly, junc_pred_index):
    edges_sa = []
    for ci, cc in enumerate(junc_pred_index):
        junc_id = np.asarray(cc).reshape(-1, 1)
        edge_id = np.concatenate((junc_id, np.roll(junc_id, -1)), axis=1)
        edges_sa.extend(edge_id)
    edges_sa = np.asarray(edges_sa)
    return edges_sa

def junc_edge_to_poly(juncs, junc_pred_index):
    poly = []
    for ci, cc in enumerate(junc_pred_index):
        if ci == 0:
            poly.extend(juncs[cc].tolist())
            poly.append(juncs[0].tolist())
        else:
            if len(cc) > 2:
                poly.extend(juncs[np.asarray(cc)-1].tolist())
                poly.append(juncs[cc[0]-1].tolist())
    poly = np.asarray(poly)
    return poly
