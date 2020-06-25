import math
import os
import cv2
import numpy as np

CLIP_LEN, RESIZE_HEIGHT, CROP_SIZE = 16, 128, 112


def process_video(video_name):
    print('Preprocess {}'.format(video_name))
    # initialize a VideoCapture object to read video data into a numpy array
    capture = cv2.VideoCapture(video_name)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    # make sure the preprocessed video has at least CLIP_LEN frames
    extract_frequency = 4
    if frame_count // extract_frequency <= CLIP_LEN:
        extract_frequency -= 1
        if frame_count // extract_frequency <= CLIP_LEN:
            extract_frequency -= 1
            if frame_count // extract_frequency <= CLIP_LEN:
                extract_frequency -= 1

    count, i, retaining = 0, 0, True
    while count < frame_count and retaining:
        retaining, frame = capture.read()
        if frame is None:
            continue

        if count % extract_frequency == 0:
            resize_height = RESIZE_HEIGHT
            resize_width = math.floor(frame_width / frame_height * resize_height)
            # make sure resize width >= crop size
            if resize_width < CROP_SIZE:
                resize_width = RESIZE_HEIGHT
                resize_height = math.floor(frame_height / frame_width * resize_width)

            frame = cv2.resize(frame, (resize_width, resize_height))

            save_name, attention = os.path.splitext(video_name)
            if not os.path.exists(save_name):
                os.makedirs(save_name)
            cv2.imwrite(os.path.join(save_name, '0000{}.jpg'.format(str(i))), frame)
            i += 1
        count += 1

    # release the VideoCapture once it is no longer needed
    capture.release()


data_dir = 'data/Fight/Fight-dataset-2020'
for item in open(os.path.join(data_dir, 'train_split.txt'), 'r'):
    name, _, _, _, _ = item.strip().split()
    process_video(os.path.join(data_dir, 'videos/', name))
    print(name + 'has done')

for item in open(os.path.join(data_dir,'val_split.txt'), 'r'):
    name, _, _, _, _ = item.strip().split()
    process_video(os.path.join(data_dir, 'videos/', name))
    print(name + 'has done')

# frames = sorted([os.path.join('test/Y_9SG3DD_tg_000003_000013', img) for img in os.listdir('test/Y_9SG3DD_tg_000003_000013')])
imgname = [int(os.path.splitext(img)[0]) for img in os.listdir('test/Y_9SG3DD_tg_000003_000013')]
imgname.sort()
for i in imgname:
    print('0000' + str(i) + '.jpg')

# for i, frame_name in enumerate(frames):
#     print(frame_name)
