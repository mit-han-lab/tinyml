import os
import shutil
import numpy as np
import tarfile

dataset_path = '~/dataset/food101'
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


def test_data():
    test_path = os.path.join(dataset_path, 'val')
    os.makedirs(test_path, exist_ok=True)
    img2id = np.genfromtxt(os.path.join(dataset_path, 'val.txt'), dtype=str)
    print(img2id.shape)
    
    for id_ in range(img2id.shape[0]):
        original_path = os.path.join(dataset_path, img2id[id_, 0][:-1])
        label = int(img2id[id_, 1])  # Label starts with 0
        
        target_path = '%s/val/%d/image_%05d.jpg' % (dataset_path, label, id_)
        
        sub_path = os.path.join(test_path, str(label))
        os.makedirs(sub_path, exist_ok=True)
        
        shutil.move(original_path, target_path)


def train_data():
    train_path = os.path.join(dataset_path, 'train')
    os.makedirs(train_path, exist_ok=True)
    img2id = np.genfromtxt(os.path.join(dataset_path, 'train.txt'), dtype=str)
    print(img2id.shape)
    
    for id_ in range(img2id.shape[0]):
        original_path = os.path.join(dataset_path, img2id[id_, 0][:-1])
        label = int(img2id[id_, 1])  # Label starts with 0
        target_path = '%s/train/%d/image_%05d.jpg' % (dataset_path, label, id_)
        
        sub_path = os.path.join(train_path, str(label))
        os.makedirs(sub_path, exist_ok=True)
        
        shutil.move(original_path, target_path)


def main():
    download_file('http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz')
    tarfile.open(os.path.join(dataset_path, 'food-101.tar.gz')).extractall(path=dataset_path)
    os.system('mv %s %s' % (os.path.join(dataset_path, 'food-101/images'), dataset_path))
    shutil.rmtree(os.path.join(dataset_path, 'food-101'))
    os.remove(os.path.join(dataset_path, 'food-101.tar.gz'))

    download_file('https://hanlab18.mit.edu/tools/image_dataset_formats/food_101/train.txt')
    download_file('https://hanlab18.mit.edu/tools/image_dataset_formats/food_101/val.txt')

    test_data()
    train_data()

    shutil.rmtree(os.path.join(dataset_path, 'images'))
    os.remove(os.path.join(dataset_path, 'train.txt'))
    os.remove(os.path.join(dataset_path, 'val.txt'))


if __name__ == '__main__':
    main()
