import os
import sys
from glob import glob
import subprocess
from tqdm import tqdm

videos = glob("/home/jack/Documents/Code/RAW_eval/synthetic_avatars/**/*.mp4", recursive=True)
watermark_root = 'outputs/RAW/watermarked2'
gt_root = 'outputs/RAW/gt2'

if not os.path.exists(watermark_root):
    os.makedirs(watermark_root)
if not os.path.exists(gt_root):
    os.makedirs(gt_root)

# Copy original videos to gt folder
for video in tqdm(videos):
    model = os.path.basename(os.path.dirname(video))
    name = os.path.basename(video).split('.')[0]
    name = f"{model}_{name}"
    gt_cmd = f"cp {video} {gt_root}/{name}.mp4"
    print(gt_cmd)
    subprocess.run(gt_cmd, shell=True)

# Add the watermark to the videos
for video in tqdm(videos):
    model = os.path.basename(os.path.dirname(video))
    name = os.path.basename(video).split('.')[0]
    name = f"{model}_{name}"
    watermark_cmd = f"python inference_av.py --input {video} --output_dir {watermark_root} --name {name} --crop_to_face"
    print(watermark_cmd)
    #subprocess.run(watermark_cmd, shell=True)

# Run attacks
attacked_root = 'outputs/RAW/attacked2'
cmd = f"python /home/jack/Documents/Code/avatarWMAttacks/video_attack.py {watermark_root} {attacked_root}"
print(cmd)
#subprocess.run(cmd, shell=True)

exit()


# Evaluate the attacks
attack_results = {}
attacked_videos = os.listdir(attacked_root)
for attacked_video in tqdm(attacked_videos, desc='Evaluating attacked videos'):

    try:
        model = attacked_video.split('_')[0]
        name = attacked_video.split('_')[1]
        name = name.replace('.mp4', '')
        name, attack = name.split('++')
        gt_path = os.path.join(watermark_root, f"{model}_{name}.txt")
        evaluate_cmd = f"python inference_av.py --detect --input {attacked_root}/{attacked_video} --gt_path {gt_path} --video_only --crop_to_face"
        print(evaluate_cmd)
        output = subprocess.run(evaluate_cmd, shell=True, stdout=subprocess.PIPE)
        output = str(output.stdout).split('\\n')
        video_acc = float(output[-2].split(' ')[-1])

        print(f"Attack: {attack} - Video: {name} - Accuracy: {video_acc}")

        if attack not in attack_results:
            attack_results[attack] = []
        attack_results[attack].append(video_acc)
    except Exception as e:
        continue


# Print the results
for attack, accs in attack_results.items():
    print(f"Attack: {attack} - Average Accuracy: {sum(accs) / len(accs)}")
    

    