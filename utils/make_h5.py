import glob

import cv2
import h5py
import os
from tqdm import tqdm
import numpy as np



def video2h5(video_path, h5_path, segment_len=16):

    h = h5py.File(h5_path, 'a')
    videos = glob.glob(video_path+'/*')

    for i in range(len(videos)):
        key = videos[i].split('\\')[-1]
        frames = glob.glob(videos[i]+'/*')
        count = 0

        for j in range(int(len(frames)//segment_len)):
            temp_frames = []
            temp_key = key + '-{0:06d}'.format(j)

            for k in range(int(segment_len)):
                frame = cv2.imread(frames[count])
                _, frame = cv2.imencode('.JPEG', frame)
                frame = np.array(frame).tostring()
                temp_frames.append(frame)
                count = count + 1

            temp_frames = np.asarray(temp_frames)
            h.create_dataset(temp_key, data=temp_frames, chunks=True)

    print('finished')


video_path = './total_frames'
h5_path = 'frames.h5'
video2h5(video_path, h5_path)

# def Video2ImgH5(video_dir, h5_path, train_list, segment_len=16, max_vid_len=2000):
#     # not multi-thread, may take time
#     h = h5py.File(h5_path, 'a')
#
#     for path in tqdm(train_list):
#         vc = cv2.VideoCapture(os.path.join(video_dir, path))
#         vid_len = vc.get(cv2.CAP_PROP_FRAME_COUNT)
#         for i in tqdm(range(int(vid_len//segment_len))):
#             tmp_frames = []
#             key = path.split('/')[-1].split('.')[0]+'-{0:06d}'.format(i)
#
#             for j in range(segment_len):
#                 ret, frame = vc.read()
#                 _, frame = cv2.imencode('.JPEG', frame)
#                 frame = np.array(frame).tostring()
#                 if ret:
#                     tmp_frames.append(frame)
#                 else:
#                     print('Bug Reported!')
#                     exit(-1)
#             tmp_frames = np.asarray(tmp_frames)
#             h.create_dataset(key, data=tmp_frames, chunks=True)
#         print(path)
#
#     print('finished!')
#


# if __name__ == '__main__':
#     video_dir = '/data0/jiachang/Anomaly-Videos/'
#     h5_file_path = '/data0/jiachang/UCFCrime-Frames-16.h5'
#     txt_path = '/data0/jiachang/Weakly_Supervised_VAD/Datasets/Anomaly_Train.txt'
#     train_list = []
#
#     with open(txt_path, 'r')as f:
#         paths = f.readlines()
#         for path in paths:
#             ano_type = path.strip().split('/')[0]
#             if 'Normal' in ano_type:
#                 path = 'Normal/'+path.strip().split('/')[-1]
#                 # continue
#             train_list.append(path.strip())
#     print(train_list)
#
#     Video2ImgH5(video_dir, h5_file_path, train_list, segment_len=16)
