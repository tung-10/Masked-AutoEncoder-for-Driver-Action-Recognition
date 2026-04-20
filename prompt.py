import torch.nn as nn
import clip
import torch
import numpy as np

prefix = "A person "

list_of_texts = [
    f'{prefix}is sitting still, waiting, not performing any specific action.',
    f'{prefix}is driving safely, keeping both hands on the steering wheel and eyes on the road.',
    f'{prefix}is reaching down to change gears with one hand while the other remains on the steering wheel.',
    f'{prefix}is turning their head to check the mirrors around the vehicle.',
    f'{prefix}is picking up a bottle and drinking water while seated.',
    f'{prefix}is raising one hand to touch or adjust their hair.',
    f'{prefix}is turning their head and body sideways to talk to a passenger.',
    f'{prefix}is looking down and using their phone while seated.',
    f'{prefix}is picking up a phone call, raising their hand to hold the phone to their ear.',
    f'{prefix}is nodding their head up and down repeatedly.',
    f'{prefix}is reaching sideways with one arm to grab something out of reach.',
]

list_of_labels = [
    'Waiting',
    'Driving safely',
    'Changing gears',
    'Checking mirrors',
    'Drinking water',
    'Touching hair',
    'Talking to passengers',
    'Checking the phone',
    'Picking up a call',
    'Nodding',
    'Reaching sideways',
]

# Safety labels for reference
safety_labels = {
    'Waiting': 'Safe',
    'Driving safely': 'Safe',
    'Changing gears': 'Safe',
    'Checking mirrors': 'Safe',
    'Drinking water': 'Unsafe',
    'Touching hair': 'Unsafe',
    'Talking to passengers': 'Unsafe',
    'Checking the phone': 'Unsafe',
    'Picking up a call': 'Unsafe',
    'Nodding': 'Unsafe',
    'Reaching sideways': 'Unsafe',
}

def text_prompt_openai_random():
    return clip.tokenize(list_of_texts)

def create_logits(x1, x2, logit_scale=10):
    print("x1:", x1.shape)
    print("x2:", x2.shape)
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    print(logits_per_x1.shape)
    print(logits_per_x2.shape)

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

class TextCLIP(nn.Module):
    def __init__(self, model_name, device):
        super(TextCLIP, self).__init__()
        self.model_name = model_name
        self.device = device
        self.load_model_clip()

    def load_model_clip(self):
        # ViT-L/14 outputs 768-dim features, matching ViT-Base visual encoder
        self.model_clip, self.preprocess = clip.load("ViT-L/14", self.device)
        self.model_clip = self.model_clip.float()

    def tokenize_list(self):
        token_list = np.array(text_prompt_openai_random())
        token_list = torch.tensor(token_list)
        return token_list

    def forward(self, text):
        return self.model_clip.encode_text(text)
