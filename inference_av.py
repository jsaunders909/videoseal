# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test with:
    python inference_av.py --input assets/videos/1.mp4 --output_dir outputs/
    python inference_av.py --detect --input outputs/1.mp4
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torchaudio
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import videoseal
from videoseal.utils.display import save_video_audio_to_mp4
import mediapipe as mp
import numpy as np

try:
    from audioseal import AudioSeal
    is_audioseal_installed = True
except ImportError:
    is_audioseal_installed = False
    print("The audioseal package is not installed. Please install it to run this script with audio watermarking.")

import cv2

def quick_write(video, name='test.png'):
    if len(video.shape) == 4:
        video = video[0]
    video = video.flip(0)
    cv2.imwrite(name, (video.permute(1, 2, 0).detach().cpu().numpy()))

def crop_faces_in_video(video_tensor, det, left_margin=0.05, right_margin=0.05, top_margin=0.1, bottom_margin=0.1):
    """ 
    Crop the faces in the video tensor using the mediapipe face mesh detector.
    """

    # Get the landmarks of the face
    bboxes = []
    for frame in video_tensor:
        frame = frame.permute(1, 2, 0).cpu().numpy()
        results = det.process(frame)

        if results.multi_face_landmarks is None:
            bboxes.append(bboxes[-1])

        lmks = results.multi_face_landmarks[0].landmark
        lmks = np.array(
            [[lmk.x * frame.shape[1], lmk.y * frame.shape[0]] for lmk in lmks]
        )
        bbox = [
            int(np.min(lmks[:, 0])),
            int(np.min(lmks[:, 1])),
            int(np.max(lmks[:, 0])),
            int(np.max(lmks[:, 1])),
        ]

        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        bbox[0] -= int(left_margin * w)
        bbox[1] -= int(top_margin * h)
        bbox[2] += int(right_margin * w)
        bbox[3] += int(bottom_margin * h)

        if w > h:
            diff = w - h
            bbox[1] -= diff // 2
            bbox[3] += diff // 2
        else:
            diff = h - w
            bbox[0] -= diff // 2
            bbox[2] += diff // 2
        bboxes.append(bbox)

    fixed_bbox = [min([bbox[0] for bbox in bboxes]), min([bbox[1] for bbox in bboxes]), max([bbox[2] for bbox in bboxes]), max([bbox[3] for bbox in bboxes])]
    
    w, h = fixed_bbox[2] - fixed_bbox[0], fixed_bbox[3] - fixed_bbox[1]
    if w > h:
        diff = w - h
        fixed_bbox[1] -= diff // 2
        fixed_bbox[3] += diff // 2
    else:
        diff = h - w
        fixed_bbox[0] -= diff // 2
        fixed_bbox[2] += diff // 2
    
    #bboxes = [fixed_bbox for _ in bboxes]
    
    # Crop
    cropped_video = []
    for frame, bbox in zip(video_tensor, bboxes):
        frame = frame[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
        frame = F.interpolate(frame.unsqueeze(0), (256, 256), mode="bilinear", align_corners=False)
        cropped_video.append(frame)
    quick_write(torch.cat(cropped_video, dim=0))
    return torch.cat(cropped_video, dim=0), bboxes
            
def return_face_bboxes(original_video_tensor, new_video_tensor, bboxes, weight=3.0):
    output_video_tensor = original_video_tensor.clone()
    for i, bbox in enumerate(bboxes):
        frame = new_video_tensor[i]
        size = (bbox[3] - bbox[1], bbox[2] - bbox[0])
        frame = F.interpolate(frame.unsqueeze(0), size, mode="bilinear", align_corners=False)
        output_video_tensor[i, :, bbox[1]:bbox[3], bbox[0]:bbox[2]] += frame[0] * weight
    return output_video_tensor



def main(args):

    # Check if the audioseal package is installed
    if not is_audioseal_installed and not args.video_only:
        raise ImportError("""Please install the audioseal package to run this script with audio watermarking.  
                        Use the --video_only flag to perform only video watermarking.  
                        Or install the package using 'pip install audioseal'.""")

    # Create the output directory and path
    os.makedirs(args.output_dir, exist_ok=True)
    if args.name is None:
        args.output = os.path.join(args.output_dir, os.path.basename(args.input))
    else:
        args.output = os.path.join(args.output_dir, args.name + ".mp4")

    # Load the VideoSeal model
    video_model = videoseal.load("videoseal")
    video_model.eval()
    video_model.to(device)

    # Read the video and convert to tensor format
    video, audio, info = torchvision.io.read_video(args.input, output_format="TCHW")
    # Normalize the video frames to the range [0, 1] and trim to 1 second

    if args.crop_to_face:
        video_orig = video.clone()
        det = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1,
            static_image_mode=True,
        )
        video, bbox = crop_faces_in_video(video, det)
        video_orig = video_orig.float() / 255.0

    if not args.video_only:
        assert "audio_fps" in info, "The input video must contain an audio track. Simply refer to the main videoseal inference code if not."
        sample_rate = info["audio_fps"]
        audio = audio.float()

    fps = info["video_fps"]
    video = video.float() / 255.0

    if not args.detect:

        # Perform watermark embedding on video
        with torch.no_grad():
            outputs = video_model.embed(video, is_video=True)

        # Extract the results
        video_w = outputs["imgs_w"]  # Watermarked video frames
        video_msgs = outputs["msgs"]  # Watermark messages

        if args.crop_to_face:
            delta = video_w - video
            video_w = return_face_bboxes(video_orig, delta, bbox)

        if not args.video_only:
            # Resample the audio to 16kHz for watermarking
            audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(audio)

            # If the audio has more than one channel, average all channels to 1 channel
            if audio_16k.shape[0] > 1:
                audio_16k_mono = torch.mean(audio_16k, dim=0, keepdim=True)
            else:
                audio_16k_mono = audio_16k

            # Add batch dimension to the audio tensor
            audio_16k_mono_batched = audio_16k_mono.unsqueeze(0)

            # Load the AudioSeal model
            audio_model = AudioSeal.load_generator("audioseal_wm_16bits")

            # Get the watermark for the audio
            with torch.no_grad():
                audio_msg = torch.randint(
                    0,
                    2,
                    (audio_16k_mono_batched.shape[0], audio_model.msg_processor.nbits),
                    device=audio_16k_mono_batched.device,
                )
                watermark = audio_model.get_watermark(
                    audio_16k_mono_batched, 16000, message=audio_msg
                )

            # Embed the watermark in the audio
            audio_16k_w = audio_16k_mono_batched + watermark

            # Remove batch dimension from the watermarked audio tensor
            audio_16k_w = audio_16k_w.squeeze(0)

            # If the original audio had more than one channel, duplicate the watermarked audio to all channels
            if audio_16k.shape[0] > 1:
                audio_16k_w = audio_16k_w.repeat(audio_16k.shape[0], 1)

            # Resample the watermarked audio back to the original sample rate
            audio_w = torchaudio.transforms.Resample(16000, sample_rate)(audio_16k_w)
        else:
            audio_w = audio
            audio_msg = None

        # Save the watermarked video and audio
        save_video_audio_to_mp4(
            video_tensor=video_w,
            audio_tensor=audio_w,
            fps=int(fps),
            audio_sample_rate=int(sample_rate),
            output_filename=args.output,
        )

        # save the watermark messages
        with open(args.output.replace(".mp4", ".txt"), "w") as f:
            msgs_str = "".join([str(msg.item()) for msg in video_msgs[0]])
            if audio_msg is not None:
                msgs_str += "_" + "".join([str(msg.item()) for msg in audio_msg[0]])
            f.write(msgs_str)


        print(f"encoded message: \n Audio: {audio_msg} \n Video {video_msgs[0]}")

    else:
        # Detect watermarks in the video
        with torch.no_grad():
            msg_extracted = video_model.extract_message(video)
        print(f"Extracted message from video: {msg_extracted}")

        if not args.video_only:
            if len(audio.shape) == 2:
                audio = audio.unsqueeze(0)  # batchify

            # if stereo convert to mono
            if audio.shape[1] > 1:
                audio = torch.mean(audio, dim=1, keepdim=True)

            # Load the AudioSeal detector model
            detector = AudioSeal.load_detector("audioseal_detector_16bits")

            # Detect watermarks in the audio
            with torch.no_grad():
                result, message = detector.detect_watermark(
                    torchaudio.transforms.Resample(sample_rate, 16000)(audio), 16000
                )
            print(f"Detection result for audio: {result}")
            print(f"Extracted message from audio: {message}")

        if args.gt_path:
            with open(args.gt_path, "r") as f:
                msgs = f.read().split("_")
                video_msg = torch.tensor([int(i) for i in msgs[0]])
                audio_msg = torch.tensor([int(i) for i in msgs[1]])

            print(f"Original message: \n Audio: {audio_msg} \n Video {video_msg}")
            video_acc = torch.sum(video_msg == msg_extracted).item() / video_msg.shape[0]
            print(f"Accuracy: \n Video {video_acc}")

            if not args.video_only:
                audio_acc = torch.sum(audio_msg == message).item() / audio_msg.shape[0]
                print(f"Accuracy: \n Audio {audio_acc}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Video and Audio Watermarking")
    parser.add_argument( "--input", type=str, required=True, help="Path to the input mp4 file")
    parser.add_argument("--output_dir",type=str,required=False, default="outputs", help="Output directory")
    parser.add_argument("--video_only", action="store_true", help="Watermark only the video, not the audio")
    parser.add_argument("--detect", action="store_true", help="Detect watermarks in the output video and audio")
    parser.add_argument("--gt_path", type=str, help="Path to the ground truth message file")
    parser.add_argument("--name", type=str, help="Name of the output file")
    parser.add_argument("--crop_to_face", action="store_true", help="Crop the video to the face")
    args = parser.parse_args()

    main(args)