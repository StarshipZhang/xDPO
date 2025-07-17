# Diffusion Preference Alignment via Attenuated  Kullback–Leibler Regularization

# Model Checkpoints

The below are initialized with StableDiffusion v1.5 models and trained as described in the paper (replicable with launchers/ scripts assuming 2 4090 GPUs,)
The StableDiffusion1.5 Model Checkpoints can be found at this anyonymous [link](https://mega.nz/folder/BO9lGDJa#ORq-W39B6QJsVPUBcACIEA)


# Structure
- `requirements.txt` Basic pip requirements

## Reproduce Results
Additional Dependencies:
`pip install -r requirements.txt`
hpsv2: https://github.com/tgxs002/HPSv2
ImageReward: https://github.com/THUDM/ImageReward


## Generate images 
```shell

python  PartiPrompts_sd15.py   --unet_model_name  <pretrained_model_name folder>
python  HPSv2_sd15.py --unet_model_name   <pretrained_model_name folder>
python  PartiPrompts_sd15.py   --unet_model_name  "mhdang/dpo-sd1.5-text2image-v1"
python  HPSv2_sd15.py --unet_model_name   "mhdang/dpo-sd1.5-text2image-v1"

"jacklishufan/diffusion-kto"
"DwanZhang/SePPO"
"mhdang/dpo-sd1.5-text2image-v1"
```

## Evaluation
To obtain the scores reported in this study, execute the following shell commands: 
```shell
python   HPSv2_Eval.py   --img_path   <image output directory>
python   PickScore.py   --img_path  <image output directory>
python   Aesthetics_score.py  --img_path  <image output directory>
python   image_reward.py   --img_path  <image output directory>
python   clip_score.py   --img_path <image output directory>

python   HPSv2_Eval.py   --img_path   'checkpoint-6000'
python   PickScore.py   --img_path  'checkpoint-6000'
python   Aesthetics_score.py  --img_path  '_checkpoint-6000'
python   image_reward.py   --img_path  'checkpoint-6000'
python   clip_score.py   --img_path 'checkpoint-6000'

```




