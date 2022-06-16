import os
import argparse
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

import torch
from torchvision import datasets, transforms
import tensorflow as tf

from mcunet.model_zoo import download_tflite

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # use only cpu for tf-lite evaluation

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--net_id', type=str, help='net id of the model')
# dataset args.
parser.add_argument('--dataset', default='imagenet', type=str)
parser.add_argument('--data-dir', default='/dataset/imagenet/val',
                    help='path to validation data')
parser.add_argument('--batch-size', type=int, default=256,
                    help='input batch size for training')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers')

args = parser.parse_args()


def get_val_dataset(resolution):
    # NOTE: we do not use normalization for tf-lite evaluation; the input is normalized to 0-1
    kwargs = {'num_workers': args.workers, 'pin_memory': False}
    if args.dataset == 'imagenet':
        val_transform = transforms.Compose([
            transforms.Resize(int(resolution * 256 / 224)),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ])
    elif args.dataset == 'vww':
        val_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),  # if center crop, the person might be excluded
            transforms.ToTensor(),
        ])
    else:
        raise NotImplementedError
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, **kwargs)
    return val_loader


def eval_image(data):
    image, target = data
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 2, 3, 1)
    image_np = image.cpu().numpy()
    image_np = (image_np * 255 - 128).astype(np.int8)
    interpreter.set_tensor(
        input_details[0]['index'], image_np.reshape(*input_shape))
    interpreter.invoke()
    output_data = interpreter.get_tensor(
        output_details[0]['index'])
    output = torch.from_numpy(output_data).view(1, -1)
    is_correct = torch.argmax(output, dim=1).item() == target.item()
    return is_correct


if __name__ == '__main__':
    tflite_path = download_tflite(net_id=args.net_id)
    interpreter = tf.lite.Interpreter(tflite_path)
    interpreter.allocate_tensors()

    # get input & output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    resolution = input_shape[1]

    # we first cache the whole test set into memory for faster data loading
    # it can reduce the testing time from ~20min to ~2min in my experiment
    print(' * start caching the test set...', end='')
    val_loader = get_val_dataset(resolution)  # range [0, 1]
    val_loader_cache = [v for v in val_loader]
    images = torch.cat([v[0] for v in val_loader_cache], dim=0)
    targets = torch.cat([v[1] for v in val_loader_cache], dim=0)

    val_loader_cache = [[x, y] for x, y in zip(images, targets)]
    print('done.')
    print(' * dataset size:', len(val_loader_cache))

    # use multi-processing for faster evaluation
    n_thread = 32

    p = Pool(n_thread)
    correctness = []

    pbar = tqdm(p.imap_unordered(eval_image, val_loader_cache), total=len(val_loader_cache),
                desc='Evaluating...')
    for idx, correct in enumerate(pbar):
        correctness.append(correct)
        pbar.set_postfix({
            'top1': sum(correctness) / len(correctness) * 100,
        })
    print('* top1: {:.2f}%'.format(
        sum(correctness) / len(correctness) * 100,
    ))
