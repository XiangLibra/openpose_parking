import os
import scipy.stats

import time

import cv2
import numpy as np
import torch
from data_process.process_utils import resize_hm, denormalize
#from visualization.visualize_test import visualize_output_single
from .post import decode_pose, append_result

# Typical evaluation is done on multi-scale and average across all evals is taken as output
# These reduce the quantization error in the model
def eval_net( model, opts,TestImage="../data/val2017/"):
    model.eval()
    # dataset = data_loader.dataset
    scales = [1., 0.5, 0.75, 1.25, 1.5, 2.0]
    assert (scales[0]==1)
    n_scales = len(scales)
    outputs = []
    dataset_len = 2 #len(dataset)
    keypoints_list = []
    runtimes = []
    MEAN = [0.485, 0.456, 0.406],
    STD = [0.229, 0.224, 0.225]
    all_image_name=os.listdir(TestImage) # 所有影像檔名(含.jpg)，import os
    with torch.no_grad():
        start = time.time()
            
        index=0
            # imgs, heatmap_t, paf_t, ignore_mask_t, keypoints = dataset.get_imgs_multiscale(i, scales,flip=False)
        for image_name in all_image_name:
            print(image_name)
            img = cv2.imread(TestImage+image_name)
            img = img.astype('float32') / 255.
            imgs=[]

            for scale in scales:
                width, height = img.shape[1], img.shape[0]
                new_width, new_height = int(scale* width), int(scale*height)
                scaled_img = cv2.resize(img.copy(), (new_width, new_height))
                scaled_img = scaled_img[:, :, ::-1]
                scaled_img = (scaled_img - MEAN) / STD
                scaled_img = scaled_img.transpose(2, 0, 1)
                imgs.append(scaled_img)

            n_imgs = len(imgs)
            assert(n_imgs == n_scales)
            heights = list(map(lambda x: x.shape[1], imgs))
            widths = list(map(lambda x: x.shape[2], imgs))
            max_h, max_w = max(heights), max(widths)
            imgs_np = np.zeros((n_imgs, 3, max_h, max_w))
            for j in range(n_imgs):
                img = imgs[j]
                h, w = img.shape[1], img.shape[2]
                imgs_np[j,:,:h,:w] = img
            img_basic = imgs[0]

            heatmap_avg_lst = []
            paf_avg_lst = []
            print("first loop", time.time() - start)
            for j in range(0, n_imgs):
                imgs_torch = torch.from_numpy(imgs_np[j:j+1]).float().cuda()
                heatmaps, pafs = model(imgs_torch)
                heatmap = heatmaps[-1].data.cpu().numpy()[0, :, :heights[j]//8, :widths[j]//8]
                paf = pafs[-1].data.cpu().numpy()[0, :, :heights[j]//8, :widths[j]//8]
                heatmap = resize_hm(heatmap, (widths[0], heights[0]))
                paf = resize_hm(paf, (widths[0], heights[0]))
                heatmap_avg_lst += [heatmap]
                paf_avg_lst += [paf]
            heatmap_avg = sum(heatmap_avg_lst)/n_imgs
            paf_avg = sum(paf_avg_lst)/n_imgs
            print("second loop", time.time() - start)
            img_basic = denormalize(img_basic)
            param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
            canvas, to_plot, candidate, subset = decode_pose(img_basic, param, heatmap_avg, paf_avg)
            final = time.time()-start
            runtimes += [final]
            print("both loops took ", final)
#            append_result(index, subset, candidate, outputs)
            index+=1
            vis_path = os.path.join(opts.saveDir, 'viz')
            if not os.path.exists(vis_path):
                os.makedirs(vis_path)
            cv2.imwrite(vis_path+'/{}'.format(image_name), to_plot)
    print ("runtime statistics for all images")
    print(scipy.stats.describe(runtimes))
    return outputs#, dataset.indices[:dataset_len]
