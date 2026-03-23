# """Extract features for temporal action detection datasets"""
# import argparse
# import os
# import random

# import numpy as np
# import torch
# from timm.models import create_model
# from torchvision import transforms
# from torchsummary import summary

# # NOTE: Do not comment `import models`, it is used to register models
# import models  # noqa: F401
# from dataset.loader import get_video_loader


# model = create_model(
#         "vit_large_patch16_224",
#         img_size=224,
#         pretrained=False,
#         num_classes=700,
#         all_frames=16,
#         tubelet_size=2,
#         drop_path_rate=0.3,
#         use_mean_pooling=True)
# ckpt = torch.load("./weights/vit_l_hybrid_pt_800e_k700_ft.pth", map_location='cpu')
# for model_key in ['model', 'module']:
#     if model_key in ckpt:
#         ckpt = ckpt[model_key]
#         break
# model.load_state_dict(ckpt)
# model.eval()
# model.cuda()
# # print(model)
# summary(model, (3, 16, 224, 224))
import glob
import os
path = "/mnt/disk2/home/thangtran/cobot/VideoMAEv2/data/EgoGesture/clips/test"

for i in os.listdir(path):
    new = os.path.join(path, i)
    print(f"{i}: {len(os.listdir(new))}")
