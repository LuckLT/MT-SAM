# -*- coding: utf-8 -*-

"""
usage example:
python MedSAM_Inference.py -i assets/img_demo.png -o ./ --box "[95,255,190,350]"

"""

# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
from nnunetv2.MedSAM.segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
import SimpleITK as sitk
from nnunetv2.MedSAM.segment_anything.modeling import prompt_encoder
from scipy import ndimage

def point_selection(mask_sim, topk):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_values, topk_xy = mask_sim.flatten(0).topk(topk, largest=True, sorted=True)  # 指定dim为0
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy % h).unsqueeze(0)  # 使用取余操作来获取列索引
    # topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()

    # Top-last point selection
    last_values, last_xy = mask_sim.flatten(0).topk(topk, largest=False, sorted=True)  # 指定dim为0
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy % h).unsqueeze(0)  # 使用取余操作来获取列索引
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()

    return topk_xy, topk_label, last_xy, last_label

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_np, point, H, W):
    box_torch = torch.as_tensor(box_np, dtype=torch.float, device=img_embed.device)
    # 在 point[0] 前面加一个维度，使其变为 1x2x2
    point[0] = np.expand_dims(point[0], axis=0)

    # 在 point[1] 前面加一个维度，使其变为 1x2
    point[1] = np.expand_dims(point[1], axis=0)
    point_torch = [torch.from_numpy(p) for p in point]

    device = img_embed.device
    points_torch = [p.to(device) for p in point_torch]
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=points_torch,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().detach().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg
# %% load model and image
parser = argparse.ArgumentParser(
    description="run inference on testing set based on MedSAM"
)
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    default="assets/img_demo.png",
    help="path to the data folder",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    default="assets/",
    help="path to the segmentation folder",
)
parser.add_argument(
    "--box",
    type=str,
    default='[95, 255, 190, 350]',
    help="bounding box of the segmentation target",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    default="/home/litaozhao/nnUNet2/nnUNet/nnunetv2/MedSAM/work_dir/MedSAM/medsam_vit_b.pth",
    help="path to the trained model",
)
args = parser.parse_args()

device = args.device
medsam_model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
medsam_model = medsam_model.to(device)
medsam_model.eval()

# temp样本的mask路径
inf = '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset601_Loc_pro/imagesTs/'  # 需要分割的test
sim_map_path = '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset601_Loc_pro/similarity_map' # 相似度图 （384*384）需缩放到原始尺寸大小
seg = '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset601_Loc_pro/Loc_sim/' # test 定位模型的分割结果
# seg = '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset601_Loc_pro/seg_SwinUNet/' # test 定位模型的分割结果
outpath = '/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset601_Loc_pro/MedSAM_results' # 输出路径


for d in os.listdir(inf):
    if '0002.nii.gz' in d:
        # 推理图像
        inf_img = os.path.join(inf, d)
        name = d.split('_')[1].split('.')[0]
        img_ori = sitk.ReadImage(inf_img)
        spacing = img_ori.GetSpacing()
        origin = img_ori.GetOrigin()
        direction = img_ori.GetDirection()
        data_img = sitk.GetArrayFromImage(img_ori)
        data_img_1024 = transform.resize(data_img, (data_img.shape[0], 1024, 1024), order=3, preserve_range=True, anti_aliasing=True)
        # img_1024_1 = sitk.GetImageFromArray(data_img_1024)
        # sitk.WriteImage(img_1024_1, os.path.join('/home/litaozhao/nnUNet2/nnUNet/nnunetv2/nnUNetFrame/DATASET/nnUNet_raw/Dataset601_Loc_pro/resize', 'PCa_' + str(name) + '.nii.gz'))
        _, H, W = data_img.shape

        # test 定位模型分割后的结果
        segpath = os.path.join(seg, 'PCa_' + str(name) + '.nii.gz')  # test 定位模型的分割结果
        seg_img = sitk.ReadImage(segpath)
        seg_data = sitk.GetArrayFromImage(seg_img)

        sim_map = os.path.join(sim_map_path, 'PCa_' + str(name) + '_prob.nii.gz')
        sim_img = sitk.ReadImage(sim_map)
        sim_data = sitk.GetArrayFromImage(sim_img)

        IMG_Seg = np.zeros(shape=(data_img.shape))
        for i in range(data_img.shape[0]):
            f = str(i) + '.nii.gz'
            data_i = data_img[int(i), :, :]
            seg_i = seg_data[int(i), :, :]
            if seg_i.max() != 0:
                # test的slice缩放尺寸到[1024,1024]
                img_1024 = transform.resize(data_i, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(
                    np.uint8)
                img_1024 = (img_1024 - img_1024.min()) / np.clip(img_1024.max() - img_1024.min(), a_min=1e-8,
                                                                 a_max=None)  # normalize to [0, 1], (H, W, 3)

                # convert the shape to (3, H, W)
                img_1024_tensor = (torch.tensor(img_1024).float().unsqueeze(2).permute(2, 0, 1).unsqueeze(0).to(device))
                img_1024_tensor = img_1024_tensor.repeat(1, 3, 1, 1)   # [1, 3, 1024, 1024]

                # 将相似度图缩放至1024*1024

                sim_data_i = sim_data[int(i), :, :]
                sim_data_1024 = transform.resize(sim_data_i, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True).astype(np.float32)

                # 根据定位模型寻找box
                mask_np_resized = transform.resize(seg_i, (1024, 1024), order=0, preserve_range=True, anti_aliasing=True).astype(np.float32)
                B = np.argwhere(mask_np_resized)
                (ystart, xstart), (ystop, xstop) = B.min(axis=0), B.max(axis=0) + 1
                ybot, xbot, ytop, xtop = ystart, xstart, ystop, xstop
                print(ybot, xbot, ytop, xtop)

                #  截取box区域的相似度图
                sim_new = sim_data_1024[ybot:ytop, xbot:xtop]
                sim_new = torch.from_numpy(sim_new).to('cuda')
                # threshold = 0.95
                # if sim_new.max() >= threshold:
                    # 寻找box区域的相似度图中的最高响应位置 Positive-negative location prior
                print(sim_new.shape)
                topk_xy_i, topk_label_i, last_xy_i, last_label_i = point_selection(sim_new, topk=1) # topk_xy_i最高相应点的坐标， topk_label_i 最高响应的label
                print('topk_xy_i', topk_xy_i)
                # 对每个坐标点的x坐标加上xbot，y坐标加上ybot
                topk_xy_i[:, 0] += xbot
                topk_xy_i[:, 1] += ybot

                last_xy_i[:, 0] += xbot
                last_xy_i[:, 1] += ybot
                print('topk_xy_i after adding offsets', topk_xy_i)
                # y, x = get_high_response_center(sim_new, threshold=threshold)
                # coor_y = int(y + ybot)  # 纵坐标（y坐标）加上dy
                # coor_x = int(x + xbot)  # 横坐标（x坐标）加上dx
                # Obtain the target guidance for cross-attention layers
                print('topk_xy_i', topk_xy_i)
                # point = [topk_xy_i, 1]# 选择第一个点
                point = [topk_xy_i, topk_label_i]# 选择第一个点
                # topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
                # topk_label = np.concatenate([topk_label_i,last_label_i], axis=0)
                # point = [topk_xy, topk_label]# 选择第一个点
                print(point)
                # if sim_data_1024[topk_xy_i[0][1], topk_xy_i[0][0]] >= 0.7:
                with torch.no_grad():
                    image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
                    ybot, xbot, ytop, xtop = ystart-5, xstart-5, ystop+5, xstop+5
                    args.box = '[' + str(xbot) + ', ' + str(ybot) + ', ' + str(xtop) + ', ' + str(ytop) + ']'
                    box_np = np.array([[int(x) for x in args.box[1:-1].split(',')]])
                    medsam_seg = medsam_inference(medsam_model, image_embedding,box_np, point, H, W)
                    IMG_Seg[i,:,:] = medsam_seg
                # if sim_data_1024[topk_xy_i[0][1], topk_xy_i[0][0]] < 0.7:
                #     with torch.no_grad():
                #         print(seg_i.shape)
                #         # if seg_i.max() < 0.4:
                #         # seg_i[sim_data_i<0.4]=0
                #
                #         IMG_Seg[i,:,:] = seg_i

            else:
                IMG_Seg[i,:,:] = seg_i
            # 对IMG_Seg中相距较近的区域填充1，封闭起来
            # struct_element = np.ones((3, 3))
            # IMG_Seg_dilated = ndimage.binary_dilation(IMG_Seg[i, :, :], structure=struct_element)
            # IMG_Seg[i, :, :] = ndimage.binary_fill_holes(IMG_Seg_dilated).astype(np.uint8)
        Img_Seg = sitk.GetImageFromArray(IMG_Seg)
        Img_Seg.SetSpacing(spacing)
        Img_Seg.SetOrigin(origin)
        Img_Seg.SetDirection(direction)
        sitk.WriteImage(Img_Seg, os.path.join(outpath, segpath.split('/')[-1]))
