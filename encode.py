import os 
import subprocess
import torchvision
import videoseal
import torch
import cv2

def main(args):

    # Load video and normalize to [0, 1]
    video_path = args.video
    video = torchvision.io.read_video(video_path, output_format="TCHW")
    video, audio, info = video
    video = video.float() / 255.0

    # Load the message
    with open(args.watermark, "r") as f:
        msg = f.read()
        msg = torch.tensor([int(c) for c in msg])[None]

    # Load the model
    model = videoseal.load("videoseal")

    # Video Watermarking
    outputs = model.embed(video, msg[:, :96])
    video_w = outputs["imgs_w"] # the watermarked video
    msgs = outputs["msgs"] # the embedded message

    # Save the watermarked video
    output_path = args.output
    video_w = (video_w * 255).permute(0, 2, 3, 1).detach().cpu().numpy().astype("uint8")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), info["video_fps"], (video_w.shape[2], video_w.shape[1]))
    for frame in video_w:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)

    writer.release()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', type=str, help='path to video file', required=True)
    parser.add_argument('-o', '--output', type=str, help='path to output directory', required=True)
    parser.add_argument('-w', '--watermark', type=str, help='path to watermark file', required=True)

    args = parser.parse_args()
    main(args)