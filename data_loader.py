import os

import pandas as pd
from PIL import Image

import torch
from torch.utils import data
import numpy as np
import scipy.io as scio
import cv2


class VideoDataset_images_with_audio_features(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, data_dir_3D, filename_path,data_audio, transform, phase, crop_size, feature_type):
        super(VideoDataset_images_with_audio_features, self).__init__()

        self.videos_dir = data_dir
        self.filename_path = filename_path
        self.dataInfo = pd.read_csv(self.filename_path) if self.filename_path else None
        #self.video_names = self.dataInfo['Video']
        self.video_names = self._get_video_names()
        self.score = self.dataInfo['Score'] if self.dataInfo is not None else None
        self.crop_size = crop_size
        self.data_dir_3D = data_dir_3D
        self.data_audio = data_audio
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.phase = phase

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.phase == 'train':
            video_name = self.video_names.iloc[idx]
        if self.phase == 'test':
            video_name = self.video_names[idx]

        if self.score is not None:
            video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx])))
        else:
            video_score = torch.FloatTensor([-1])

        if self.phase == 'train':
            new_video_dir = "/Talking_head/images_train/"
        if self.phase == 'test':
            new_video_dir = "/Talking_head/images_test/"

        path_name = os.path.join(new_video_dir, video_name.split(".m")[0])

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        video_length_read = 8

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        video_read_index = 0
        for i in range(video_length_read):
            file_name_tar = os.path.join(path_name.split('.mp4')[0])
            file_num = len([name for name in os.listdir(file_name_tar) if os.path.isfile(os.path.join(file_name_tar, name))])
            # print(file_num)
            imge_name = os.path.join(path_name.split('.mp4')[0], 'frame_{}.png'.format(i % file_num))
            try:
                read_frame = Image.open(imge_name)
                read_frame = read_frame.convert('RGB')
                read_frame = self.transform(read_frame)
                # transformed_video[i] = read_frame
                transformed_video[video_read_index] = read_frame
                # transformed_video = transformed_video + read_frame
                video_read_index += 1
            except:
                print (imge_name)

        if video_read_index < video_length_read:
            for j in range(video_read_index, video_length_read):
                transformed_video[j] = transformed_video[video_read_index - 1]

        # for i in range(video_length_read):
        #     imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
        #     read_frame = Image.open(imge_name)
        #     read_frame = read_frame.convert('RGB')
        #     read_frame = self.transform(read_frame)
        #     transformed_video[i] = read_frame

        # read 3D features
        if self.phase == 'train':
            feature_folder_name = os.path.join(self.data_dir_3D, self.video_names.iloc[idx].split('.mp4')[0])
        if self.phase == 'test':
            feature_folder_name = os.path.join(self.data_dir_3D, self.video_names[idx].split('.mp4')[0])

        if self.feature_type == 'Slow':
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            transformed_feature = torch.zeros([video_length_read, 2048 + 256])
            for i in range(video_length_read):
                i_index = i
                feature_3D_slow = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index % file_num) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index % file_num) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

        # read audio features
        def pad_tensor(tensor, target_shape):
            pad_width = []
            for t_size, target_size in zip(tensor.shape, target_shape):
                if t_size < target_size:
                    pad_before = 0
                    pad_after = target_size - t_size
                else:
                    pad_before = 0
                    pad_after = 0

                pad_width.append((pad_before, pad_after))

            return np.pad(tensor, pad_width=pad_width, mode='constant', constant_values=0)


        if self.phase == 'train':
            feature_folder_name = os.path.join(self.data_audio, self.video_names.iloc[idx].split('.mp4')[0])
        if self.phase == 'test':
            feature_folder_name = os.path.join(self.data_audio, self.video_names[idx].split('.mp4')[0])

        audio_feature = torch.zeros([video_length_read, 30, 4, 128, 62])
        target_shape = (30, 4, 128, 62)
        for i in range(video_length_read):
            audio_feature_path = os.path.join((feature_folder_name) + '.npy')
            audio_tensor = np.load(audio_feature_path)[:30, :4, :, :]
            afeatures = pad_tensor(audio_tensor, target_shape)
            afeatures = torch.from_numpy(afeatures)
            afeatures = afeatures.squeeze()
            audio_feature[i] = afeatures

        if self.phase == 'test':
            return transformed_video, transformed_feature, audio_feature, video_name
        if self.phase == 'train':
            return transformed_video, transformed_feature, audio_feature, video_score, video_name

    def _get_video_names(self):
        if self.dataInfo is None:
            video_names = sorted(os.listdir(self.videos_dir))
            return video_names
        else:
            #return self.datainfo['Video'].tolist()
            return self.dataInfo['Video']


class VideoDataset_images_VQA_dataset_with_motion_features(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, data_dir_3D, filename_path, transform, database_name, crop_size, feature_type, exp_id,
                 state='train'):
        super(VideoDataset_images_VQA_dataset_with_motion_features, self).__init__()

        if database_name == 'KoNViD-1k':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][exp_id]
            if state == 'train':
                index = index_all[:int(n * 0.8)]
            elif state == 'val':
                index = index_all[int(n * 0.8):]

            for i in index:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][i][0])
            self.video_names = video_names
            self.score = score

        elif database_name == 'youtube_ugc':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            score = []
            index_all = dataInfo['index'][exp_id]
            if state == 'train':
                index = index_all[:int(n * 0.8)]
            elif state == 'val':
                index = index_all[int(n * 0.8):]

            for i in index:
                video_names.append(dataInfo['video_names'][i][0][0])
                score.append(dataInfo['scores'][0][i])
            self.video_names = video_names
            self.score = score

        self.crop_size = crop_size
        self.videos_dir = data_dir
        self.data_dir_3D = data_dir_3D
        self.transform = transform
        self.length = len(self.video_names)
        self.feature_type = feature_type
        self.database_name = database_name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]
        video_score = torch.FloatTensor(np.array(float(self.score[idx])))

        path_name = os.path.join(self.videos_dir, video_name_str)

        video_channel = 3

        video_height_crop = self.crop_size
        video_width_crop = self.crop_size

        if self.database_name == 'KoNViD-1k':
            video_length_read = 8
        elif self.database_name == 'youtube_ugc':
            video_length_read = 10

        transformed_video = torch.zeros([video_length_read, video_channel, video_height_crop, video_width_crop])

        for i in range(video_length_read):
            if self.database_name == 'youtube_ugc':
                imge_name = os.path.join(path_name, '{:03d}'.format(i * 2) + '.png')
            else:
                imge_name = os.path.join(path_name, '{:03d}'.format(i) + '.png')
            read_frame = Image.open(imge_name)
            read_frame = read_frame.convert('RGB')
            read_frame = self.transform(read_frame)
            transformed_video[i] = read_frame

        # read 3D features
        if self.feature_type == 'Slow':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048])
            for i in range(video_length_read):
                if self.database_name == 'KoNViD-1k':
                    i_index = i
                elif self.database_name == 'youtube_ugc':
                    i_index = int(i / 2)
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'Fast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 256])
            for i in range(video_length_read):
                if self.database_name == 'KoNViD-1k':
                    i_index = i
                elif self.database_name == 'youtube_ugc':
                    i_index = int(i / 2)
                feature_3D = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D = torch.from_numpy(feature_3D)
                feature_3D = feature_3D.squeeze()
                transformed_feature[i] = feature_3D
        elif self.feature_type == 'SlowFast':
            feature_folder_name = os.path.join(self.data_dir_3D, video_name_str)
            transformed_feature = torch.zeros([video_length_read, 2048 + 256])
            for i in range(video_length_read):
                if self.database_name == 'KoNViD-1k':
                    i_index = i
                elif self.database_name == 'youtube_ugc':
                    i_index = int(i / 2)
                feature_3D_slow = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
                feature_3D_slow = torch.from_numpy(feature_3D_slow)
                feature_3D_slow = feature_3D_slow.squeeze()
                feature_3D_fast = np.load(
                    os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
                feature_3D_fast = torch.from_numpy(feature_3D_fast)
                feature_3D_fast = feature_3D_fast.squeeze()
                feature_3D = torch.cat([feature_3D_slow, feature_3D_fast])
                transformed_feature[i] = feature_3D

        return transformed_video, transformed_feature, video_score, video_name

class VideoDataset_NR_LSVQ_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize, is_test_1080p=False):
        super(VideoDataset_NR_LSVQ_SlowFast_feature, self).__init__()
        if is_test_1080p:
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_valid']
        else:
            column_names = ['name', 'p1', 'p2', 'p3', \
                            'height', 'width', 'mos_p1', \
                            'mos_p2', 'mos_p3', 'mos', \
                            'frame_number', 'fn_last_frame', 'left_p1', \
                            'right_p1', 'top_p1', 'bottom_p1', \
                            'start_p1', 'end_p1', 'left_p2', \
                            'right_p2', 'top_p2', 'bottom_p2', \
                            'start_p2', 'end_p2', 'left_p3', \
                            'right_p3', 'top_p3', 'bottom_p3', \
                            'start_p3', 'end_p3', 'top_vid', \
                            'left_vid', 'bottom_vid', 'right_vid', \
                            'start_vid', 'end_vid', 'is_test', 'is_valid']

        dataInfo = pd.read_csv(filename_path, header=0, sep=',', names=column_names, index_col=False,
                               encoding="utf-8-sig")

        self.video_names = dataInfo['name']
        self.score = dataInfo['mos']
        self.videos_dir = data_dir
        self.transform = transform
        self.resize = resize
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names.iloc[idx]
        video_score = torch.FloatTensor(np.array(float(self.score.iloc[idx]))) / 20

        filename = os.path.join(self.videos_dir, video_name + '.mp4')

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)

        video_channel = 3

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        video_clip = int(video_length / video_frame_rate)

        video_clip_min = 8

        video_length_clip = 32

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []

        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1

        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

        video_capture.release()

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i * video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[
                                    i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])

        return transformed_video_all, video_score, video_name

class VideoDataset_NR_SlowFast_feature(data.Dataset):
    """Read data from the original dataset for feature extraction"""

    def __init__(self, data_dir, filename_path, transform, resize, database_name):
        super(VideoDataset_NR_SlowFast_feature, self).__init__()

        if database_name == 'KoNViD-1k':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            for i in range(n):
                video_names.append(dataInfo['video_names'][i][0][0])
            self.video_names = video_names

        elif database_name == 'youtube_ugc':
            dataInfo = scio.loadmat(filename_path)
            n = len(dataInfo['video_names'])
            video_names = []
            for i in range(n):
                video_names.append(dataInfo['video_names'][i][0][0])
            self.video_names = video_names

        self.transform = transform
        self.videos_dir = data_dir
        self.resize = resize
        self.database_name = database_name
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        video_name = self.video_names[idx]
        video_name_str = video_name[:-4]
        filename = os.path.join(self.videos_dir, video_name)

        video_capture = cv2.VideoCapture()
        video_capture.open(filename)
        cap = cv2.VideoCapture(filename)

        video_channel = 3

        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))

        if video_frame_rate == 0:
            video_clip = 10
        else:
            video_clip = int(video_length / video_frame_rate)

        if self.database_name == 'KoNViD-1k':
            video_clip_min = 8
        elif self.database_name == 'youtube_ugc':
            video_clip_min = 20

        video_length_clip = 32

        transformed_frame_all = torch.zeros([video_length, video_channel, self.resize, self.resize])

        transformed_video_all = []

        video_read_index = 0
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                read_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                read_frame = self.transform(read_frame)
                transformed_frame_all[video_read_index] = read_frame
                video_read_index += 1

        if video_read_index < video_length:
            for i in range(video_read_index, video_length):
                transformed_frame_all[i] = transformed_frame_all[video_read_index - 1]

        video_capture.release()

        for i in range(video_clip):
            transformed_video = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            if (i * video_frame_rate + video_length_clip) <= video_length:
                transformed_video = transformed_frame_all[
                                    i * video_frame_rate: (i * video_frame_rate + video_length_clip)]
            else:
                transformed_video[:(video_length - i * video_frame_rate)] = transformed_frame_all[i * video_frame_rate:]
                for j in range((video_length - i * video_frame_rate), video_length_clip):
                    transformed_video[j] = transformed_video[video_length - i * video_frame_rate - 1]
            transformed_video_all.append(transformed_video)

        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                transformed_video_all.append(transformed_video_all[video_clip - 1])

        return transformed_video_all, video_name_str
