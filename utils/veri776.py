# vim: expandtab:ts=4:sw=4
import os
import numpy as np
import cv2
import scipy.io as sio
import torch.utils.data as data
from PIL import Image

class VeRi776(data.Dataset):

    def __init__(self, data_dir, transform=None, dataset_name=None, has_gt=True):
        self.imgs, self.ids, self.cameras = read_file_str(data_dir)
        self.transform = transform
        self.name = dataset_name
        self.has_gt = has_gt
        self._id2label = {_id: idx for idx, _id in enumerate(np.unique(self.ids))}

    def __getitem__(self, index):

        path = self.imgs[index]
        label = self._id2label[self.ids[index]]
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)
        
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.ids)

def _parse_filename(filename):
    """Parse meta-information from given filename.

    Parameters
    ----------
    filename : str
        A Market 1501 image filename.

    Returns
    -------
    (int, int, str, str) | NoneType
        Returns a tuple with the following entries:

        * Unique ID of the individual in the image
        * Index of the camera which has observed the individual
        * Filename without extension
        * File extension

        Returns None if the given filename is not a valid filename.

    """
    filename_base, ext = os.path.splitext(filename)
    if '.' in filename_base:
        # Some images have double filename extensions.
        filename_base, ext = os.path.splitext(filename_base)
    if ext != ".jpg":
        return None
    car_id, cam_seq, frame_idx, detection_idx = filename_base.split('_')
    cam_id = int(cam_seq[1:])
    return int(car_id), cam_id, filename_base, ext


def read_file_str(image_dir):
    """Read training data to list of filenames.

    Parameters
    ----------
    dataset_dir : str
        Path to the Market 1501 dataset directory.

    Returns
    -------
    (List[str], List[int], List[int])
        Returns a tuple with the following values:

        * List of image filenames (full path to image files).
        * List of unique IDs for the individuals in the images.
        * List of camera indices.

    """
    filenames, ids, cameraids = [], [], []

    for filename in sorted(os.listdir(image_dir)):
        meta_data = _parse_filename(filename)
        if meta_data is None:
            # This is not a valid filename (e.g., Thumbs.db).
            continue
        if meta_data[0] < 0:
            continue
        filenames.append(os.path.join(image_dir, filename))
        ids.append(meta_data[0])
        cameraids.append(meta_data[1])

    return filenames, ids, cameraids