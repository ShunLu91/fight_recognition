import os
import math
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

CLIP_LEN, RESIZE_HEIGHT, CROP_SIZE = 16, 128, 112

class VideoDataset(Dataset):
    def __init__(self, file_path, root='vidoes', mode='train', clip_len=32):
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112
        self.clip_len = clip_len
        self.mode = mode
        self.root = root

        # obtain all the filenames of files inside all the class folders
        # going through each class folder one at a time
        self.fnames= []
        self.labels = []
        for item in open(file_path, 'r'):
            path, fps, cnt, r, label = item.strip().split()
            self.fnames.append(os.path.join(root, path))
            self.labels.append(int(label))

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        # buffer = self.loadvideo(self.fnames[index]) # [D, H, W, C]
        buffer = self.load_frames(self.fnames[index]) # [D, H, W, C]

        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer).type(torch.FloatTensor), self.labels[index]

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def __len__(self):
        return len(self.fnames)

    def loadvideo(self, fname):
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        # buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        buffer = []

        count = 0
        retaining = True

        # read in each frame, one at a time into the numpy buffer array
        extract_frequency = 4
        if frame_count // extract_frequency <= self.clip_len:
            extract_frequency -= 1
            if frame_count // extract_frequency <= self.clip_len:
                extract_frequency -= 1
                if frame_count // extract_frequency <= self.clip_len:
                    extract_frequency -= 1

        while count < frame_count and retaining:
            if count % extract_frequency == 0:
                retaining, frame = capture.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size
                # NOTE: strongly recommended to resize them during the download process. This script
                # will process videos of any size, but will take longer the larger the video file.
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                # buffer[count] = frame
                buffer.append(frame)
                count += 1

        # release the VideoCapture once it is no longer needed
        capture.release()

        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        # buffer = buffer.transpose((3, 0, 1, 2))

        return buffer

    def load_frames(self, file_video):
        file_dir = os.path.splitext(file_video)[0]
        frames = [int(os.path.splitext(img)[0]) for img in os.listdir(file_dir)]
        frames.sort()
        buffer = []

        for i in frames:
            img_path = os.path.join(file_dir, '0000'+str(i)+'.jpg')
            frame = np.array(cv2.imread(img_path))
            buffer.append(frame)

        return np.array(buffer).astype(np.uint8)

    def crop(self, buffer, clip_len, crop_size):
        if self.mode == 'train':
            # randomly select time index for temporal jitter
            if buffer.shape[0] > clip_len:
                time_index = np.random.randint(buffer.shape[0] - clip_len)
            else:
                time_index = 0
            # randomly select start indices in order to crop the video
            height_index = np.random.randint(buffer.shape[1] - crop_size)
            width_index = np.random.randint(buffer.shape[2] - crop_size)
            # crop and jitter the video using indexing. The spatial crop is performed on
            # the entire array, so each frame is cropped in the same location. The temporal
            # jitter takes place via the selection of consecutive frames
        else:
            # for val and test, select the middle and center frames
            if buffer.shape[0] > clip_len:
                time_index = math.floor((buffer.shape[0] - clip_len) / 2)
            else:
                time_index = 0
            height_index = math.floor((buffer.shape[1] - crop_size) / 2)
            width_index = math.floor((buffer.shape[2] - crop_size) / 2)
        buffer = buffer[time_index:time_index + clip_len, height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        # padding repeated frames to make sure the shape as same
        if buffer.shape[0] < clip_len:
            repeated = clip_len // buffer.shape[0] - 1
            remainder = clip_len % buffer.shape[0]
            buffered, reverse = buffer, True
            if repeated > 0:
                padded = []
                for i in range(repeated):
                    if reverse:
                        pad = buffer[::-1, :, :, :]
                        reverse = False
                    else:
                        pad = buffer
                        reverse = True
                    padded.append(pad)
                padded = np.concatenate(padded, axis=0)
                buffer = np.concatenate((buffer, padded), axis=0)
            if reverse:
                pad = buffered[::-1, :, :, :][:remainder, :, :, :]
            else:
                pad = buffered[:remainder, :, :, :]
            buffer = np.concatenate((buffer, pad), axis=0)
        return buffer

    def normalize(self, buffer):
        # Normalize the buffer
        # NOTE: Default values of RGB images normalization are used, as precomputed
        # mean and std_dev values (akin to ImageNet) were unavailable for Kinetics. Feel
        # free to push to and edit this section to replace them if found.
        buffer = (buffer - 128.0)/128.0
        return buffer

