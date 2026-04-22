import torch.nn as nn
import clip
import torch
import numpy as np

prefix = "A person "

list_of_texts = [
    f'{prefix}is sitting still with both hands resting, not moving.',
    f'{prefix}is driving with both hands firmly on the steering wheel, eyes forward.',
    f'{prefix}is moving one hand down to the gear shift beside the seat.', 
    f'{prefix}is briefly glancing sideways at the side mirror while keeping hands on wheel.',
    f'{prefix}is lifting a bottle or a cup to their mouth and drinking.',
    f'{prefix}is lifting one hand up to touch their hair on their head.',
    f'{prefix}is turning their head to the side to talk ơr show expression.', 
    f'{prefix}is taking a phone in hand and looking down at the screen.', 
    f'{prefix}is raising their hand to hold a ringing phone against their ear.',
    f'{prefix}is moving their head suddenly down and up.', 
    f'{prefix}is stretching one arm far to the side to reach an object.',

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
