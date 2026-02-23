import torch.nn as nn
import clip
import torch
import numpy as np

prefix = "A person "
list_of_texts = [
f'{prefix} does nothing',
f'{prefix}Raise one arm up high, make a fist, then keep the fist still, bend the arm to shoulder level, then raise it up again.'

f'{prefix}Place one hand in front of your chest. Then move the arm in the opposite direction.',

f'{prefix}Place your arms vertical and perpendicular to the ground at shoulder level when your arms are bent. Then bring it back to your chest.',

f'{prefix}Raise one hand at chest level, palm facing down, tip of the hand facing out, then press the hand down to hip level, then lift the arm up and bring it back to the original position.',

f'{prefix}Place one hand flat on your hip, tip of the hand facing outward. Then, lift the hand lying face up to chest level, then press the hand down again and return to the starting position.',

f'{prefix}Place one arm across your chest, palm facing down, tip of the hand facing the opposite hand. Then, move your arm along the chest plane at an angle of more than 120 degrees, at which time the tip of your hand points outward. Finally return your arms to the starting position.',

f'{prefix}Raise one arm overhead, holding the hand, then move the arm and keep the holding hand towards the top of the head. Finally, return your arms to the starting position.',

f'{prefix}Raise one arm horizontally across the chest, vertical and perpendicular to the ground, hand with one index finger pointing up to the sky. Then, move your arm and index finger up until your arm is at shoulder level. Finally, return to the original position.',

f'{prefix}Raise one arm, place the hand on one half of the head.',

f'{prefix}Place one arm across your chest, vertical to your body, perpendicular to the ground. Raise your index finger and thumb normally, with your index finger pointing toward the sky.',

f'{prefix}Place your arms in front of your chest, clasp your hands, spread out only your thumbs, stand up and face the sky. Hold your hands facing in front of your body.',

f'{prefix}Place your arms in front of your chest, hands open and facing each other, palm tips facing the sky.',

f'{prefix}Place your arms across your chest, your hands close together (about a few centimeters), your palms facing each other.',

f'{prefix}Place one arm in front of your chest, vertical to the body. Palms face out in front of the body, fingers clustered in a claw shape.',

f'{prefix}Place one arm in front of your chest, vertical to the body. Hands clenched.',

f'{prefix}Place your hands across your chest, arms pointing out in front of your body. The hand bends the ring finger and little finger, the middle finger contacts the thumb, and the index finger spreads out naturally.',

f'{prefix}Place one arm across the neck, the arm horizontally perpendicular to the body axis, the tip of the hand facing the opposite shoulder. The other arm places the hand in front of the chest perpendicular to the palm of the other hand, the tip of the hand facing up.',

f'{prefix}Place one hand on the opposite shoulder, arms spread out with five fingers embracing one shoulder.',

f'{prefix}Place one hand on the opposite shoulder, placing the hand close to the shoulder. Hold the 3 thumbs, ring finger and little finger, and spread the index and middle fingers.',

f'{prefix}Raise one arm in front of your chest, arm vertical to the body axis. The fingers spread out and come together at one point, the tips of the hands facing the front of the body.']

list_of_labels  = ['START',
'STOP',
'SLOWER',
'FASTER',
'DONE',
'FOLLOW ME',
'LIFT',
'HOME',
'LOOK',
'OK',
'HELP',
'AGAIN',
'PICKPART',
'DEPOSIT PART',
'INTERACTION',
'JOYSTICK',
'IDENTIFICATION',
'CHANGE',
'REPORT',]

a = ['No gesture','Start', 'Stop', 'Slower', 'Faster', 'Done', 'FollowMe', 'Lift', 'Home', 'Interaction', 'Look', 'PickPart', 'DepositPart', 'Report', 'Ok', 'Again', 'Help', 'Joystick', 'Identification', 'Change']
b = ['No gesture','START', 'STOP', 'SLOWER', 'FASTER', 'DONE', 'FOLLOW ME', 'LIFT', 'HOME', 'LOOK', 'OK', 'HELP', 'AGAIN', 'PICKPART', 'DEPOSIT PART', 'INTERACTION', 'JOYSTICK', 'IDENTIFICATION', 'CHANGE', 'REPORT']

# Convert both lists to lowercase for comparison
a_lower = [item.lower() for item in a]
b_lower = [item.lower().replace(' ', '') for item in b]

# Get the order of items in list b based on list a
# Create a dictionary with the order of items in list a
order = {item: i for i, item in enumerate(a_lower)}

# Sort list b based on the order in list a
list_of_texts_sorted = sorted(list_of_texts, key=lambda item: order.get(item.lower().replace(' ', ''), float('inf')))

def text_prompt_openai_random():
    return clip.tokenize(list_of_texts_sorted)

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
    def __init__(self, model_name, device) :
        super(TextCLIP, self).__init__()
        self.model_name = model_name
        self.device = device
        self.load_model_clip()

    def load_model_clip(self):
        self.model_clip, self.preprocess = clip.load(self.model_name, self.device)
        self.model_clip = self.model_clip.float()

    def tokenize_list(self):
        token_list = np.array(text_prompt_openai_random())
        token_list = torch.tensor(token_list)
        return token_list

    def forward(self,text):
        return self.model_clip.encode_text(text)


# text_embedding_list = model_text(token_list).float()