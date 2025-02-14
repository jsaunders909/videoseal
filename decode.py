import torch
import torchvision
import os
import argparse
import videoseal
from videoseal.utils.display import save_video_audio_to_mp4
import mediapipe as mp
import numpy as np

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the VideoSeal model
    video_model = videoseal.load("videoseal")
    video_model.eval()
    video_model.to(device)

    # Read the video and convert to tensor format
    video, audio, info = torchvision.io.read_video(args.video, output_format="TCHW")
    video = video.to(device)
    video = video.float() / 255.0

    # Detect watermarks in the video
    with torch.no_grad():
        msg_extracted = video_model.extract_message(video).squeeze().cpu().numpy()

    message_str = "".join([str(x) for x in msg_extracted])
    print(f"Extracted message from video: {message_str}")

    with open(args.output, 'w') as f:
        f.write(message_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", type=str, required=True, help="Path to the input video")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output file")
    args = parser.parse_args()
    main(args)