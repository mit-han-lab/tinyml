import os
import shutil
import numpy as np
import tarfile


dataset_path = '~/dataset/pets'
dataset_path = os.path.expanduser(dataset_path)

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)


def download_file(url, dest=None):
    if not dest:
        dest = os.path.join(dataset_path, url.split('/')[-1])
    if os.path.exists(dest):
        print('%s exists' % dest)
        return
    run_cmd = ('wget %s -O %s' % (url, dest))
    print(run_cmd)
    os.system(run_cmd)

download_file('https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz')
tarfile.open(os.path.join(dataset_path, 'images.tar.gz')).extractall(path=dataset_path)

download_file('https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz')
tarfile.open(os.path.join(dataset_path, 'annotations.tar.gz')).extractall(path=dataset_path)

# build train+val
train_val_list = os.path.join(dataset_path, 'annotations/trainval.txt')
with open(train_val_list, 'r') as fin:
    for line in fin.readlines():
        line = line[:-1].split(' ')
        file_name, class_id = line[0], int(line[1])
        file_name += '.jpg'
        src_path = os.path.join(dataset_path, 'images/%s' % file_name)

        target_folder = os.path.join(dataset_path, 'train/%d' % class_id)
        os.makedirs(target_folder, exist_ok=True)
        target_path = os.path.join(target_folder, file_name)
        shutil.move(src_path, target_path)
        print('Moving %s to %s' % (src_path, target_path))


# build test
test_list = os.path.join(dataset_path, 'annotations/test.txt')
with open(test_list, 'r') as fin:
    for line in fin.readlines():
        line = line[:-1].split(' ')
        file_name, class_id = line[0], int(line[1])
        file_name += '.jpg'
        src_path = os.path.join(dataset_path, 'images/%s' % file_name)

        target_folder = os.path.join(dataset_path, 'val/%d' % class_id)
        os.makedirs(target_folder, exist_ok=True)
        target_path = os.path.join(target_folder, file_name)
        shutil.move(src_path, target_path)
        print('Moving %s to %s' % (src_path, target_path))

os.remove(os.path.join(dataset_path, 'images.tar.gz'))
os.remove(os.path.join(dataset_path, 'annotations.tar.gz'))

shutil.rmtree(os.path.join(dataset_path, 'images'))
shutil.rmtree(os.path.join(dataset_path, 'annotations'))

