
import os

import argparse
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json

from utils.clip_utils import Selector

# load model
device = "cuda"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path, cache_dir='./ckpt')
model_AutoMode = AutoModel.from_pretrained(model_pretrained_name_or_path, cache_dir='./ckpt').eval().to(device)


def calc_probs(prompt, images):
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        # embed
        image_embs = model_AutoMode.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = model_AutoMode.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = model_AutoMode.logit_scale.exp() * (text_embs @ image_embs.T)[0]

    return scores



environ_root = os.environ.get('HPS_ROOT')
root_path = os.path.expanduser('~/.cache/hpsv2') if environ_root == None else environ_root

# from . import utils

# utils.download_benchmark_prompts()
data_path = os.path.join(root_path, 'datasets/benchmark')
if not os.path.exists(root_path):
    os.makedirs(root_path)

    
parser = argparse.ArgumentParser()

parser.add_argument(
    '--img_path', 
    default="/root/autodl-tmp/DiffusionDPO/hpsv2_output/fine_sd15_checkpoint_checkpoint-800",
)
args = parser.parse_args()   
    
 
    
meta_dir = data_path
img_path = args.img_path

style_list = os.listdir(img_path)
model_id = img_path.split('/')[-1]

score = {}

model_dict = {}
model_name = "ViT-H-14"
precision = 'amp'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 20


class BenchmarkDataset(Dataset):
    def __init__(self, meta_file, image_folder, transforms, tokenizer):
        self.transforms = transforms
        self.image_folder = image_folder
        self.tokenizer = tokenizer
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


model_dict = {}
model_name = "ViT-H-14"
precision = 'amp'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def initialize_model():
    if not model_dict:
        model, preprocess_train, preprocess_val = create_model_and_transforms(
            model_name,
            None,
            precision=precision,
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False
        )
        model_dict['model'] = model
        model_dict['preprocess_val'] = preprocess_val


def collate_eval(batch):
    images = torch.stack([sample[0] for sample in batch])
    captions = torch.cat([sample[1] for sample in batch])
    return images, captions


# from HPSv2.hpsv2.evaluation import BenchmarkDataset
# from HPSv2.hpsv2.evaluation import initialize_model
# from HPSv2.hpsv2.evaluation import collate_eval
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

initialize_model()
model = model_dict['model']
preprocess_val = model_dict['preprocess_val']
tokenizer = get_tokenizer(model_name)
selector = Selector(device)
score = {}
score[model_id] = {}
for style in style_list:
    score[model_id][style] = []
    image_folder = os.path.join(img_path, style)
    meta_file = os.path.join(meta_dir, f'{style}.json')
    dataset = BenchmarkDataset(meta_file, image_folder, preprocess_val, tokenizer)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_eval)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    a = 0
    with torch.no_grad():
        for i, batch in enumerate(dataset):
            img_pa, texts = batch
            # images = images.to(device=device, non_blocking=True)
            # texts = texts.to(device=device, non_blocking=True)

            a = selector.score(img_pa,texts)

            score[model_id][style].extend(a)



for model_id, data in score.items():
    all_score = []
    for style , res in data.items():
        res = torch.tensor(res)

        avg_score = [torch.mean(res[i:i+80]) for i in range(0, len(res), 80)]
        avg_score_stack = torch.stack(avg_score)
        all_score.extend(res)
        all_score_stack = torch.stack(all_score)
        # tensor_stack = torch.stack(all_score)
        print(model_id, '{:<15}'.format(style), '{:.3f}'.format(torch.mean(avg_score_stack)), '\t', '{:.4f}'.format(torch.mean(avg_score_stack)))
    print(model_id, '{:<15}'.format('Average'), '{:.3f}'.format(torch.mean(all_score_stack)), '\t')
