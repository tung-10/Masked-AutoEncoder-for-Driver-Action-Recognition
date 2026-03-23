#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import pickle
import time
import torch
import tqdm
import cv2
import os 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import numpy as np
import matplotlib.pyplot as plt
import util_eval

ID_MAP = {
    "01_011_01_0_front": 11,
    "01_011_01_0_rear": 11,
    "01_012_01_0_front": 12,
    "01_012_01_0_rear": 12,
    "01_013_01_0_front": 13,
    "01_013_01_0_rear": 13,
    "01_014_01_0_front": 14,
    "01_014_01_0_rear": 14,
    "01_015_01_0_front": 15,
    "01_015_01_0_rear": 15,
    "01_016_01_0_front": 16,
    "01_016_01_0_rear": 16,
    "01_017_01_0_front": 17,
    "01_017_01_0_rear": 17,
    "01_018_01_0_front": 18,
    "01_018_01_0_rear": 18,
}

def map_filename_to_id(filename):
    basename = os.path.basename(str(filename))
    name_no_ext = os.path.splitext(basename)[0]
    return ID_MAP.get(name_no_ext, filename)

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
                ,[256,0,192],[250,100,150],[128,128,256]]

cmap = np.asarray(label_colours) / 255.0
ignore_file = './error_arm.csv'
def key_sort(file):
    # basename = os.path.basename(file)
    first = str(file).split("_")[0][1:]
    last = str(file).split(".")[0][-1]
    number = int(first + last)
    return number
def read_ignore_file(file):
    data = pd.read_csv(file, header=None)
    list_start = [round(i/30) for i in data.iloc[:, 2].values.tolist()]
    list_subject = data.iloc[:, 0].values.tolist()
    subject = list(map(key_sort, list_subject))
    return subject, list_start # video_id

def process_overlap(data, name_vid, ignore_list_subject=None, ignore_list_start=None):
    data = data.sort_values(by=["start"]).reset_index(drop=True)
    for j in range(len(data)-1):
        for i in range(j+1, len(data)):
            if data.loc[i, "start"] < data.loc[j, "end"]:
                data.loc[i, "end"] = 0
                data.loc[i, "start"] = 0
                # print(data.loc[i])
        # if data.loc[j+1, "start"] - data.loc[j, "end"] < 0:
        #     data.loc[j+1, "end"] = 0
        #     data.loc[j+1, "start"] = 0
            # print(data.loc[j+1])
    # duplicate_labels = data['label'][data['label'].duplicated()].unique()
    # for i in duplicate_labels:
    #     for j in range(len(data)-1):
    #         label_a = int(data.loc[j, "label"])
    #         if label_a == i and j>=1:
    #             pre_end = int(data.loc[j-1, 'end'])
    #             former_start = int(data.loc[j, "start"])
    #             if pre_end == former_start:
    #                 data.loc[j, "end"] = 0
    #                 data.loc[j, "start"] = 0
    if not(ignore_list_subject is None) and name_vid in ignore_list_subject: # bỏ trong gesture bị lỗi
        start = ignore_list_start[ignore_list_subject.index(name_vid)]
        for j in range(len(data)-1):
            if abs(start - data.loc[j, 'start'])<=2: # int(data.loc[j, "label"]) in [11, 13] and 
                data.loc[j, "end"] = 0
                data.loc[j, "start"] = 0

    data = data[data["end"]!=0]            
    return data

def merge_and_remove(data, merge_threshold=16):
    df_total = pd.DataFrame([[0, 0, 0, 0]], columns=["video_id", "label", "start", "end"])
    data = data.reset_index(drop=True)
    data = data.sort_values(by=["video_id", "label"])
    # ignore_list_subject, ignore_list_start = read_ignore_file(ignore_file)
    for i in data["video_id"].unique():
        data_video = data[data["video_id"]==i]
        list_label = data_video["label"].unique()
        vid_all = pd.DataFrame([[0, 0, 0, 0]], columns=["video_id", "label", "start", "end"])
        for label in list_label:

            data_video_label = data_video[data_video["label"]== label]
            data_video_label = data_video_label.reset_index()
            data_video_label = data_video_label.sort_values(by=["start"])

            for j in range(len(data_video_label)-1):

                if data_video_label.loc[j+1, "start"] - data_video_label.loc[j, "end"] <= merge_threshold:
                    data_video_label.loc[j+1, "start"] = data_video_label.loc[j, "start"]
                    data_video_label.loc[j, "end"] = 0
                    data_video_label.loc[j, "start"] = 0

            vid_all = pd.concat([vid_all, data_video_label], ignore_index=True)
            
        vid_all = vid_all[vid_all["end"]!=0]
        print("vid", i)
        vid_all = process_overlap(vid_all, i)
        # vid_all = process_overlap(vid_all, i, ignore_list_subject=None, ignore_list_start=None)
        df_total = pd.concat([df_total, vid_all], ignore_index=True)
    df_total = df_total[df_total["end"]!=0]
    min_len = 3

    df_total["length"] = df_total["end"] - df_total["start"]

    df_total = df_total[df_total["length"] >= min_len]

    df_total = df_total.drop(columns=["length"])
    df_total = df_total.drop(columns=['index'])
    df_total = df_total.sort_values(by=["video_id", "start"])
    return df_total


def general_submission(data, without_post = False):
    # data = pd.read_csv(filename, sep=" ", header=None)
    # data_filtered = data[data["label"] != 0]
    data_filtered = data.copy()
    data_filtered["start"] = data["start"].map(lambda x: int(float(x)))
    data_filtered["end"] = data["end"].map(lambda x: int(float(x)))
    data_filtered = data_filtered.sort_values(by=["video_id","label"])
    if not without_post:
        results = merge_and_remove(data_filtered, merge_threshold=2)
        return results
    else:
        results = merge_and_remove(data_filtered, merge_threshold=0)
        return results


def topk_by_partition(input, k, axis=None, ascending=False):
    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis) # k non-sorted indices
    input = np.take_along_axis(input, ind, axis=axis) # k non-sorted values

    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind_part, axis=axis) 
    return ind, val


def get_classification(sequence_class_prob):
    classify=[[x,y] for x,y in zip(np.argmax(sequence_class_prob, axis=1),np.max(sequence_class_prob, axis=1))]
    labels_index = np.argmax(sequence_class_prob, axis=1) #returns list of position of max value in each list.
    probs= np.max(sequence_class_prob, axis=1)  # return list of max value in each  list.
    return labels_index, probs

def activity_localization(prob_sq, action_threshold):
    action_idx, action_probs = get_classification(prob_sq)
    threshold = np.mean(action_probs)
    action_tag = np.zeros(action_idx.shape)
    # action_tag[action_probs >= threshold] = 1
    action_tag[action_probs >= action_threshold] = 1
    # print('action_tag', action_tag)
    activities_idx = []
    startings = []
    endings = []

    for i in range(len(action_tag)):
        if action_tag[i] ==1:
            activities_idx.append(action_idx[i])
            start = i
            end = i+1
            startings.append(start)
            endings.append(end)
    return activities_idx, startings, endings

def smoothing(x, k=2):
    ''' Applies a mean filter to an input sequence. The k value specifies the window
    size. window size = 2*k
    '''
    l = len(x)
    s = np.arange(-k, l - k)
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1
    y = np.zeros(x.shape)
    for i in range(l):
        y[i] = np.mean(x[s[i]:e[i]], axis=0)
    return y

def gaussian_smoothing(x, k=3):
    ''' 
    Applies a Gaussian filter to an input sequence. 
    The k value specifies the window size (kernel_size = 2*k + 1)
    
    Args:
        x: input sequence of shape (length, num_classes)
        k: half window size (kernel_size = 2*k + 1)
    
    Returns:
        y: smoothed sequence of same shape as x
    '''
    l = len(x)
    border = k
    kernel_size = 2 * k + 1
    
    # Handle both 1D and 2D inputs
    if len(x.shape) == 1:
        # 1D case
        width = l
        num_classes = 1
        x = x.reshape(-1, 1)
    else:
        # 2D case (length, num_classes)
        width = l
        num_classes = x.shape[1]
    
    y = np.zeros(x.shape, dtype=np.float32)
    
    # Process each class channel separately
    for c in range(num_classes):
        hm = x[:, c].copy()
        origin_max = np.max(hm)
        
        # Skip if all zeros
        if origin_max == 0:
            y[:, c] = hm
            continue
        
        # Add padding
        dr = np.zeros(width + 2 * border, dtype=np.float32)
        dr[border: -border] = hm
        
        # Create 1D Gaussian kernel
        # sigma = k / 2 for smoother results
        sigma = max(0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8, 1.0)  # OpenCV default
        kernel_1d = cv2.getGaussianKernel(kernel_size, sigma)
        
        # Apply Gaussian filter
        dr_filtered = cv2.filter2D(dr.reshape(-1, 1), -1, kernel_1d)[:, 0]
        
        # Remove padding
        hm_filtered = dr_filtered[border: -border]
        
        # Normalize to preserve original max value
        if np.max(hm_filtered) > 0:
            hm_filtered *= origin_max / np.max(hm_filtered)
        
        y[:, c] = hm_filtered
    
    # Return same shape as input
    if num_classes == 1:
        y = y.reshape(-1)
    
    return y

def gauss_smoothing(x, k=1):
    ''' Applies a mean filter to an input sequence. The k value specifies the window
    size. window size = 2*k
    '''
    l = len(x)

    s = np.arange(-k, l - k)
    e = np.arange(k, l + k)
    s[s < 0] = 0
    e[e >= l] = l - 1

    f = np.zeros(k*2, dtype=np.float32)
    total = 0
    for i in range(-k, k):
        f[i+k] = np.exp(-(i/2)**2)
        total += f[i+k]
    f = f / total
    f = f.reshape(-1, 1)

    y = np.zeros(x.shape)
    for i in range(l):
        if e[i] - s[i] < 2*k:
            y[i] = np.mean(x[s[i]:e[i]], axis=0)
        else:
            y[i] = np.sum(x[s[i]:e[i]]*f, axis=0)
    return y

def compute_os_score(ground_truth, prediction, threshold):
    ground_truth_gbvn = ground_truth.groupby('video_id')
    label = ground_truth["label"].unique()
    scores = []
    for idx, this_pred in prediction.iterrows():
        video_id = this_pred['video_id']
        try:
            this_gt = ground_truth_gbvn.get_group(video_id)
            this_gt = this_gt.reset_index()
            tiou_arr = util_eval.segment_iou(this_pred[["start", "end"]].values, this_gt[["start", "end"]].values, threshold)
            scores += [item for item in tiou_arr if item > 0]
        except:
            print("Video {} gt has no {} action".format(video_id, label))
    return scores

# class_names = ['Start', 'Stop', 'Slower', 'Faster', 'Done', 'FollowMe', 'Lift', 'Home', 'Interaction', \
#               'Look', 'PickPart', "PositionPart", 'Report', "Ok", "Again", "Help", "Joystick", "Identification", 'Change']

class_names = [ "Smoking", "Calling", "Texting", "Drinking", "Yawning", "Jocking"]
class_map = dict([(str(i+1), class_names[i]) for i in range(len(class_names))])

def load_k_fold_probs(file):
    probs = []
    with open(file, "rb") as fp:
        vmae_16x4_probs = pickle.load(fp)
    probs.append(vmae_16x4_probs)
    return probs

def get_file_to_id(file):
    metadata = {}
    with open(file, 'rb') as f:
        data = pickle.load(f)
        # print(data)
        for key in data.keys():
            # print(key)
            metadata[key] = int(f"{key.split('_')[0][1:]}{key.split('_')[-1]}")

    return metadata

def softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def fusion_geometric_mean(front, rear):
    """Geometric mean - balance giữa 2 views"""
    epsilon = 1e-10
    fused = np.sqrt((front + epsilon) * (rear + epsilon))
    # Re-normalize
    return fused / (fused.sum(axis=-1, keepdims=True) + 1e-10)

def fusion_harmonic_mean(front, rear):
    """Harmonic mean - penalize low confidence predictions"""
    epsilon = 1e-10
    fused = 2.0 / (1.0/(front + epsilon) + 1.0/(rear + epsilon))
    return fused / (fused.sum(axis=-1, keepdims=True) + 1e-10)

def fusion_product_softmax(front, rear):
    """Product rồi softmax lại"""
    product = front * rear
    # Softmax
    return softmax_np(product)

def fusion_log_sum_exp(front, rear):
    """Log-sum-exp trick - numerically stable"""
    epsilon = 1e-10
    log_front = np.log(front + epsilon)
    log_rear = np.log(rear + epsilon)
    
    # Average in log space
    log_avg = (log_front + log_rear) / 2.0
    
    # Convert back
    fused = np.exp(log_avg)
    return fused / (fused.sum(axis=-1, keepdims=True) + 1e-10)

def fusion_attention_weighted(front, rear):
    """Attention-based fusion - dynamic weighting per timestep"""
    # Compute confidence scores (entropy-based)
    def entropy(probs):
        epsilon = 1e-10
        return -np.sum(probs * np.log(probs + epsilon), axis=-1, keepdims=True)
    
    ent_front = entropy(front)
    ent_rear = entropy(rear)
    
    # Lower entropy = higher confidence = higher weight
    # Inverse entropy as weight
    max_ent = np.log(front.shape[-1])  # max possible entropy
    conf_front = (max_ent - ent_front) / max_ent
    conf_rear = (max_ent - ent_rear) / max_ent
    
    # Normalize weights
    total_conf = conf_front + conf_rear + 1e-10
    w_front = conf_front / total_conf
    w_rear = conf_rear / total_conf
    
    # Weighted fusion
    return w_front * front + w_rear * rear

def fusion_temporal_aware(front, rear):
    """
    Fusion có xét đến temporal coherence
    """
    T, C = front.shape
    fused = np.zeros_like(front)
    
    for t in range(T):
        # Context window
        start_t = max(0, t - 2)
        end_t = min(T, t + 3)
        
        # Temporal consistency score
        front_temp_var = np.var(front[start_t:end_t], axis=0)
        rear_temp_var = np.var(rear[start_t:end_t], axis=0)
        
        # Lower variance = more consistent = higher weight
        w_front = 1.0 / (front_temp_var + 1e-3)
        w_rear = 1.0 / (rear_temp_var + 1e-3)
        
        # Normalize
        total = w_front + w_rear
        w_front = w_front / total
        w_rear = w_rear / total
        
        # Fuse
        fused[t] = w_front * front[t] + w_rear * rear[t]
    
    return fused

def fusion_temporal_per_class_weight(front, rear, 
                                     class_weights_rear=None,
                                     window_size=5):
    """
    Mỗi class có weight riêng cho rear view
    Ví dụ: Drinking dễ nhìn từ rear hơn → weight cao
    """
    T, C = front.shape
    
    if class_weights_rear is None:
        # Default: rear view tốt hơn cho các class này
        class_weights_rear = np.ones(C) * 0.5  # baseline 50-50
        
        # Customize per class (dựa trên domain knowledge)
        # 0: background, 1: Smoking, 2: Calling, 3: Texting, 
        # 4: Drinking, 5: Yawning, 6: Jocking
        class_weights_rear[1] = 0.6  # Smoking: rear 60%
        class_weights_rear[2] = 0.5  # Calling: 50-50
        class_weights_rear[3] = 0.55 # Texting: rear 55%
        class_weights_rear[4] = 0.65 # Drinking: rear 65% (dễ thấy từ rear)
        class_weights_rear[5] = 0.45 # Yawning: front 55% (cần thấy mặt)
        class_weights_rear[6] = 0.6  # Jocking: rear 60%
    
    class_weights_front = 1.0 - class_weights_rear
    
    fused = np.zeros_like(front)
    
    for t in range(T):
        # Temporal window
        start_t = max(0, t - window_size // 2)
        end_t = min(T, t + window_size // 2 + 1)
        
        # Temporal Variance
        front_var = np.var(front[start_t:end_t], axis=0)
        rear_var = np.var(rear[start_t:end_t], axis=0)
        
        # Temporal weight
        w_temp_front = 1.0 / (front_var + 1e-3)
        w_temp_rear = 1.0 / (rear_var + 1e-3)
        
        # Normalize temporal weight
        total_temp = w_temp_front + w_temp_rear + 1e-10
        w_temp_front = w_temp_front / total_temp
        w_temp_rear = w_temp_rear / total_temp
        
        # Combine class weight (70%) + temporal weight (30%)
        w_front = 0.7 * class_weights_front + 0.3 * w_temp_front
        w_rear = 0.7 * class_weights_rear + 0.3 * w_temp_rear
        
        # Normalize final
        total = w_front + w_rear + 1e-10
        w_front = w_front / total
        w_rear = w_rear / total
        
        # Fuse
        fused[t] = w_front * front[t] + w_rear * rear[t]
    
    return fused

def main():
    file_pickle_front = './checkpoints_front2/utcda_view.pkl'
    # _FILENAME_TO_ID = get_file_to_id(file_pickle)
    localization = []

    # load k-fold probs
    k_fold_front = load_k_fold_probs(file_pickle_front)
    
    start_time = time.time()

    for vid in k_fold_front[0].keys():
        # stack k-fold
        all_front = np.stack([
            np.array(list(map(np.array, fold[vid])))
            for fold in k_fold_front
        ])  # [K, T, C]

        avg_front = np.mean(all_front, axis=0)  # [T, C]
        
        prob_seq = avg_front
        prob_seq_smooth = smoothing(prob_seq, k=4)

        activities_idx, startings, endings = activity_localization(
            prob_seq_smooth, action_threshold=0.4
        )

        for label, s, e in zip(activities_idx, startings, endings):
            start = s * 15 / 20
            end   = e * 15 / 20
            localization.append([vid, label, start, end])

    rough_loc = pd.DataFrame(localization, columns=["video_id", "label", "start", "end"])
    prediction = general_submission(rough_loc, without_post=False)
    prediction["video_id"] = prediction["video_id"].apply(map_filename_to_id)
    print('time', time.time() - start_time)
    prediction.to_csv("pred_front.csv",
                      columns=["video_id", "label", "start", "end"], index=False)

    # load pred file
    CLASS_NUM = 7
    gt = pd.read_csv("test_grt.csv")
    gt_eval = gt[gt["label"] != 0]
    pred_eval = prediction[prediction["label"] != 0]
    M, N = len(gt_eval), len(pred_eval)
    gt_by_label = gt.groupby("label")
    pred_by_label = prediction.groupby("label")

    scores = []
    for label in range(1, CLASS_NUM):
        try:
            ground_truth_class = gt_by_label.get_group(label).reset_index(drop=True)
            prediction_class = pred_by_label.get_group(label).reset_index(drop=True)   
            scores += compute_os_score(ground_truth_class, prediction_class, threshold =60)
        except:
            continue
    print("Total Action:", M)
    print("True Positive:", len(scores))
    print("False Positive:", N-len(scores))
    print("False Negtive:", M-len(scores))
    print("score", sum(scores) / (M+N-len(scores)))
    print("PRECISON AVG", sum(scores)/len(scores))
    print("Recall AVG", sum(scores)/M)

main()