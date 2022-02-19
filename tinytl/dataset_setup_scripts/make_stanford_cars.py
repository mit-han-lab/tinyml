import os
import scipy.io as io
import shutil
import tarfile


dataset_path = '~/dataset/stanford_car'
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


def get_data():
    download_file('http://ai.stanford.edu/~jkrause/car196/cars_train.tgz')
    download_file('http://ai.stanford.edu/~jkrause/car196/cars_test.tgz')
    download_file('https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz')
    download_file('http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat')    
    tarfile.open(os.path.join(dataset_path, 'cars_train.tgz')).extractall(path=dataset_path)
    tarfile.open(os.path.join(dataset_path, 'cars_test.tgz')).extractall(path=dataset_path)
    tarfile.open(os.path.join(dataset_path, 'car_devkit.tgz')).extractall(path=dataset_path)


def test_data():
    test_path = os.path.join(dataset_path, 'val')
    os.makedirs(test_path, exist_ok=True)
    a = io.loadmat(os.path.join(dataset_path, 'cars_test_annos_withlabels.mat'))
    b = a['annotations'][0]
    with open(os.path.join(dataset_path, 'cars_test.txt'), 'w'):
        for t in b:
            outstr = os.path.join(dataset_path, 'cars_test/%s' % t[5][0])
            class_id = t[4][0][0]
            class_path = os.path.join(test_path, '%d' % class_id)
            if not os.path.exists(class_path):
                os.makedirs(class_path, exist_ok=True)
            target_path = os.path.join(class_path, t[5][0])
            shutil.move(outstr, target_path)
            print(outstr, class_id, 'to', target_path)


def train_data():
    train_path = os.path.join(dataset_path, 'train')
    os.makedirs(train_path, exist_ok=True)
    a = io.loadmat(os.path.join(dataset_path, 'devkit/cars_train_annos.mat'))
    b = a['annotations'][0]
    with open(os.path.join(dataset_path, 'cars_train.txt'), 'w'):
        for t in b:
            outstr = os.path.join(dataset_path, 'cars_train/%s' % t[5][0])
            class_id = t[4][0][0]
            class_path = os.path.join(train_path, '%d' % class_id)
            if not os.path.exists(class_path):
                os.makedirs(class_path, exist_ok=True)
            target_path = os.path.join(class_path, t[5][0])
            shutil.move(outstr, target_path)
            print(outstr, class_id, 'to', target_path)


def clear():
    # rm other files
    for file_path in [
        'cars_test', 'cars_train', 'devkit', 'car_devkit.tgz', 'cars_test.tgz', 'cars_test.txt',
        'cars_test_annos_withlabels.mat', 'cars_train.tgz', 'cars_train.txt',
    ]:
        file_path = os.path.join(dataset_path, file_path)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            shutil.rmtree(file_path)


def main():
    get_data()
    test_data()
    train_data()
    clear()


if __name__ == '__main__':
    main()
