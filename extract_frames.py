from PIL import Image
from torchvision import transforms
import os
import cv2
import argparse
import pandas as pd
import scipy.io as scio

def extract_frames(input_video, frames_dir, video_length_read):
    cap =cv2.VideoCapture(input_video)

    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

    trans = transforms.Resize(520)

    video_read_index = 0
    frame_idx = 0
    video_length_min = 3

    for i in range(video_length):
        has_frames, frame = cap.read()
        if has_frames:
            # 1s 1frame
            if (video_read_index < video_length_read) and (frame_idx % video_frame_rate == 0):

                read_frame = Image.fromarray(cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB))
                read_frame = trans(read_frame)
                read_frame.save(os.path.join(frames_dir, 'frame_{}.png'.format(video_read_index)))
                video_read_index += 1

            frame_idx += 1

    # 不足video_length_min的，取最后一帧重复
    if video_read_index < video_length_min:
        for i in range(video_read_index, video_length_min):
            read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            read_frame = trans(read_frame)
            read_frame.save(os.path.join(frames_dir, 'frame_{}.png'.format(i)))

    if frame_idx == 0:
        print ('Frame extraction failed', input_video)
    cap.release()

    return 1

if __name__ == '__main__':

    #Info = pd.read_csv("D:/dataset/Talking_head/thqa_ntire_train.csv")
    #vids = Info.Video
    vids_dir = "D:/dataset/Talking_head/test/"
    video_names = sorted(os.listdir(vids_dir))
    frames_dir = "D:/dataset/Talking_head/images_test/"

    for vid in video_names:
        vid = vid.split()[0]
        # print(vid)
        vid_path = os.path.join(vids_dir, vid)
        frame_path = os.path.join(frames_dir, vid.split(".m")[0])

        os.makedirs(frame_path, exist_ok=True)

        print(vid_path, frame_path)
        extract_frames(vid_path, frame_path, 10)
