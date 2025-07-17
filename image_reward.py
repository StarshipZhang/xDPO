import os
import torch
import ImageReward as reward
from torch.utils.data import Dataset, DataLoader
import argparse
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from utils.clip_utils import Selector



class BenchmarkDataset(Dataset):
    def __init__(self, meta_file, image_folder):

        self.image_folder = image_folder

        self.open_image = Image.open
        with open(meta_file, 'r') as f:
            prompts = json.load(f)
        used_prompts = []
        files = []
        for idx, prompt in enumerate(prompts):
            filename = os.path.join(self.image_folder, f'{idx:05d}.jpg')
            if os.path.exists(filename):
                used_prompts.append(prompt)
                files.append(filename)
            else:
                print(f"missing image for prompt: {prompt}")
        self.prompts = used_prompts
        self.files = files

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        images = self.open_image(img_path)
        caption = self.prompts[idx]

        return img_path, caption





model = reward.load("ImageReward-v1.0")
parser = argparse.ArgumentParser()

parser.add_argument(
    '--img_path',
    default="/root/autodl-tmp/DiffusionDPO/hpsv2_output/1111_checkpoint-2400",
)
args = parser.parse_args()

environ_root = os.environ.get('HPS_ROOT')
root_path = os.path.expanduser('~/.cache/hpsv2') if environ_root == None else environ_root
data_path = os.path.join(root_path, 'datasets/benchmark')
meta_dir = data_path
img_path = args.img_path

style_list = os.listdir(img_path)
model_id = img_path.split('/')[-1]

model_dict = {}
model_name = "ViT-H-14"

score = {}
score[model_id] = {}
for style in style_list:
    score[model_id][style] = []
    image_folder = os.path.join(img_path, style)
    meta_file = os.path.join(meta_dir, f'{style}.json')
    dataset = BenchmarkDataset(meta_file, image_folder)
    a = 0
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            img_pa, texts = batch
            # images = images.to(device=device, non_blocking=True)
            # texts = texts.to(device=device, non_blocking=True)
            # a = ImageReward_model.score(texts, img_pa)
            rewards = model.score(texts, img_pa)
            score[model_id][style].extend([rewards])


for model_id, data in score.items():
    all_score = []
    for style, res in data.items():
        res = torch.tensor(res)

        avg_score = [torch.mean(res[i:i + 80]) for i in range(0, len(res), 80)]
        avg_score_stack = torch.stack(avg_score)
        all_score.extend(res)
        all_score_stack = torch.stack(all_score)
        # tensor_stack = torch.stack(all_score)
        print(model_id, '{:<15}'.format(style), '{:.3f}'.format(torch.mean(avg_score_stack)), '\t',
              '{:.4f}'.format(torch.mean(avg_score_stack)))
    print(model_id, '{:<15}'.format('Average'), '{:.3f}'.format(torch.mean(all_score_stack)), '\t')

