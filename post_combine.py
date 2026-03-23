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

from postprocessing import merge_and_remove, map_filename_to_id, general_submission, load_k_fold_probs, smoothing, activity_localization, compute_os_score

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

def softmax_np(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def fusion_multiple(front, rear):
    product = front * rear
    # Softmax
    return softmax_np(product)

def fusion_max(front, rear):
    maximum = np.maximum(front, rear)
    return softmax_np(maximum)

def fusion_average(front, rear):
    return (front + rear)/2

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

def main():
    file_pickle_front = './checkpoints_front2/utcda_view.pkl'
    file_pickle_rear = './checkpoints_rear2/utcda_view.pkl'
    # _FILENAME_TO_ID = get_file_to_id(file_pickle)
    localization = []

    # load k-fold probs
    k_fold_front = load_k_fold_probs(file_pickle_front)
    k_fold_rear  = load_k_fold_probs(file_pickle_rear)

    for vid in k_fold_front[0].keys():
        # stack k-fold
        all_front = np.stack([
            np.array(list(map(np.array, fold[vid])))
            for fold in k_fold_front
        ])  # [K, T, C]

        all_rear = np.stack([
            np.array(list(map(np.array, fold[vid.replace("front", "rear")])))
            for fold in k_fold_rear
        ])  # [K, T, C]

        avg_front = np.mean(all_front, axis=0)
        avg_rear  = np.mean(all_rear, axis=0)  
        
        prob_seq = (avg_front + avg_rear) / 2
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
    prediction.to_csv("pred_fused_front_rear.csv",
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