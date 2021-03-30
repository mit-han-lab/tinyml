# -*- coding: utf-8 -*
"""This module is served as torchvision.datasets to load CUB200-2011.

CUB200-2011 dataset has 11,788 images of 200 bird species. The project page
is as follows.
    http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
- Images are contained in the directory data/cub200/raw/images/,
  with 200 subdirectories.
- Format of images.txt: <image_id> <image_name>
- Format of train_test_split.txt: <image_id> <is_training_image>
- Format of classes.txt: <class_id> <class_name>
- Format of image_class_labels.txt: <image_id> <class_id>

This file is modified from:
    https://github.com/vishwakftw/vision.
"""


import os
import pickle
import numpy as np
import PIL.Image
import shutil
import requests

import torch.utils.data as data


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


class CUB200(data.Dataset):
    """CUB200 dataset.

    Args:
        _root, str: Root directory of the dataset.
        _train, bool: Load train/test data.
        _transform, callable: A function/transform that takes in a PIL.Image
            and transforms it.
        _target_transform, callable: A function/transform that takes in the
            target and transforms it.
        _train_data, list of np.ndarray.
        _train_labels, list of int.
        _test_data, list of np.ndarray.
        _test_labels, list of int.
    """
    def __init__(self, root, _train=True, transform=None, target_transform=None,
                 download=False):
        """Load the dataset.

        Args
            root, str: Root directory of the dataset.
            train, bool [True]: Load train/test data.
            transform, callable [None]: A function/transform that takes in a
                PIL.Image and transforms it.
            target_transform, callable [None]: A function/transform that takes
                in the target and transforms it.
            download, bool [False]: If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        self._root = os.path.expanduser(root)  # Replace ~ by the complete dir
        os.makedirs(self._root, exist_ok=True)
        self._train = _train
        self._transform = transform
        self._target_transform = target_transform

        self._download()
        self._extract()

    def _download(self):
        """Download and uncompress the tar.gz file from a given URL.

        Args:
            url, str: URL to be downloaded.
        """
        import six.moves
        import tarfile

        raw_path = os.path.join(self._root, 'raw')
        processed_path = os.path.join(self._root, 'processed')
        if not os.path.isdir(raw_path):
            os.mkdir(raw_path, mode=0o775)
        if not os.path.isdir(processed_path):
            os.makedirs(processed_path, exist_ok=True)

        # Downloads file.
        fpath = os.path.join(self._root, 'raw/CUB_200_2011.tgz')
        download_file_from_google_drive(id='1hbzc_P1FuxMkcabkgn9ZKinBwW683j45', destination=fpath)

        # Extract file.
        cwd = os.getcwd()
        tar = tarfile.open(fpath, 'r:gz')
        os.chdir(os.path.join(self._root, 'raw'))
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def _extract(self):
        """Prepare the data for train/test split and save onto disk."""
        image_path = os.path.join(self._root, 'raw/CUB_200_2011/images/')
        # Format of images.txt: <image_id> <image_name>
        id2name = np.genfromtxt(os.path.join(
            self._root, 'raw/CUB_200_2011/images.txt'), dtype=str)
        # Format of train_test_split.txt: <image_id> <is_training_image>
        id2train = np.genfromtxt(os.path.join(
            self._root, 'raw/CUB_200_2011/train_test_split.txt'), dtype=int)

        for id_ in range(id2name.shape[0]):
            src_image_path = os.path.join(image_path, id2name[id_, 1])
            label = int(id2name[id_, 1][:3]) - 1  # Label starts with 0

            if id2train[id_, 1] == 1:
                target_path = os.path.join(self._root, 'train')
                os.makedirs(target_path, exist_ok=True)
                target_path = os.path.join(target_path, id2name[id_, 1])
            else:
                target_path = os.path.join(self._root, 'val')
                os.makedirs(target_path, exist_ok=True)
                target_path = os.path.join(target_path, id2name[id_, 1])
            folder = '/'.join(target_path.split('/')[:-1])
            os.makedirs(folder, exist_ok=True)
            shutil.move(src_image_path, target_path)
            print('(%s, %s): Move from %s to %s' % (id2name[id_, 1], label, src_image_path, target_path))

dataset_path = '~/dataset/cub200'
dataset_path = os.path.expanduser(dataset_path)

train = CUB200(dataset_path, _train=True, download=True, transform=None)
test = CUB200(dataset_path, _train=False, download=True, transform=None)
shutil.rmtree(os.path.join(dataset_path, 'raw'))
shutil.rmtree(os.path.join(dataset_path, 'processed'))
