#!/usr/bin/env python
import os
import tarfile
from scipy.io import loadmat
import shutil


dataset_path = '~/dataset/flowers102'
dataset_path = os.path.expanduser(dataset_path)


def download_file(url, dest=None):
    if not dest:
        dest = os.path.join(dataset_path, url.split('/')[-1])
    run_cmd = ('wget %s -O %s' % (url, dest))
    print(run_cmd)
    os.system(run_cmd)
    # urllib.urlretrieve(url, dest)


# Download the Oxford102 flowersset into the current directory
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)

if not os.path.exists(os.path.join(dataset_path, '102flowers.tgz')):
    print('Downloading images...')
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz')
    tarfile.open(os.path.join(dataset_path, '102flowers.tgz')).extractall(path=dataset_path)

    print('Downloading image labels...')
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat')

    print('Downloading train/test/valid splits...')
    download_file('http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat')

# Read .mat file containing training, testing, and validation sets.
setid = loadmat(os.path.join(dataset_path, 'setid.mat'))

# The .mat file is 1-indexed, so we subtract one to match Caffe's convention.
idx_train = setid['trnid'][0]
idx_test = setid['tstid'][0]
idx_valid = setid['valid'][0]

# Read .mat file containing image labels.
image_labels = loadmat(os.path.join(dataset_path, 'imagelabels.mat'))['labels'][0]

train_path = os.path.join(dataset_path, 'train')
os.makedirs(train_path, exist_ok=True)
for i in range(1, 103):
    sub_path = os.path.join(train_path, '%d' % i)
    os.makedirs(sub_path, exist_ok=True)

for idx in idx_train:
    category = image_labels[idx - 1]
    original_path = '%s/jpg/image_%05d.jpg' % (dataset_path, idx)
    target_path = '%s/train/%d/image_%05d.jpg' % (dataset_path, category, idx)
    shutil.move(original_path, target_path)

for idx in idx_valid:
    category = image_labels[idx - 1]
    original_path = '%s/jpg/image_%05d.jpg' % (dataset_path, idx)
    target_path = '%s/train/%d/image_%05d.jpg' % (dataset_path, category, idx)
    shutil.move(original_path, target_path)

path = os.path.join(dataset_path, 'val')
os.makedirs(path, exist_ok=True)
for i in range(1, 103):
    sub_path = os.path.join(path, '%d' % i)
    os.makedirs(sub_path, exist_ok=True)

for idx in idx_test:
    category = image_labels[idx - 1]
    original_path = '%s/jpg/image_%05d.jpg' % (dataset_path, idx)
    target_path = '%s/val/%d/image_%05d.jpg' % (dataset_path, category, idx)
    shutil.move(original_path, target_path)


# rm other files
for to_remove in ['102flowers.tgz', 'imagelabels.mat', 'jpg', 'setid.mat']:
    file_path = os.path.join(dataset_path, to_remove)
    if os.path.isfile(file_path):
        os.remove(file_path)
    else:
        shutil.rmtree(file_path)
