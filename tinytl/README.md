# TinyTL: Reduce Activations, Not Trainable Parameters for Efficient On-Device Learning [[website]](https://hanlab.mit.edu/projects/tinyml/tinyTL/)

```BibTex
@inproceedings{
  cai2020tinytl,
  title={TinyTL: Reduce Memory, Not Parameters for Efficient On-Device Learning},
  author={Cai, Han and Gan, Chuang and Zhu, Ligeng and Han, Song},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## On-Device Learning, not Just Inference
<p align="center">
    <img src="https://hanlab.mit.edu/projects/tinyml/tinyTL/figures/on-device-learning.png" width="80%" />
</p>

## Activation is the Main Bottleneck, not Parameters
<p align="center">
    <img src="https://hanlab.mit.edu/projects/tinyml/tinyTL/figures/acitvation-is-the-bottleneck.png" width="70%" />
</p>

## Tiny Transfer Learning
<p align="center">
    <img src="https://hanlab.mit.edu/projects/tinyml/tinyTL/figures/tinyTL.png" width="70%" />
</p>

## Transfer Learning Results
<p align="center">
    <img src="https://hanlab.mit.edu/projects/tinyml/tinyTL/figures/tinytl_results_8.png" width="80%" />
</p>

## Combining with Batch Size 1 Training
<p align="center">
    <img src="https://hanlab.mit.edu/projects/tinyml/tinyTL/figures/tinytl_batch1.png" width="80%" />
</p>

## Data Preparation
To set up the datasets, please run `bash make_all_datasets.sh` under the folder **dataset_setup_scripts**.

## Requirement
* Python 3.6+
* Pytorch 1.4.0+

## How to Run Transfer Learning Experiments
To run transfer learning experiments, please first set up the datasets and then run **tinytl_fgvc_train.py**. 
Scripts are available under the folder **exp_scripts**.

## TODO

- [ ] Add system support for TinyTL 


## Related Projects

[MCUNet: Tiny Deep Learning on IoT Devices](https://arxiv.org/abs/2007.10319) (NeurIPS'20, spotlight)

[Once for All: Train One Network and Specialize it for Efficient Deployment](https://arxiv.org/abs/1908.09791) (ICLR'20)

[ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware](https://arxiv.org/pdf/1812.00332.pdf) (ICLR'19)

[AutoML for Architecting Efficient and Specialized Neural Networks](https://ieeexplore.ieee.org/abstract/document/8897011) (IEEE Micro)

[AMC: AutoML for Model Compression and Acceleration on Mobile Devices](https://arxiv.org/pdf/1802.03494.pdf) (ECCV'18)

[HAQ: Hardware-Aware Automated Quantization](https://arxiv.org/pdf/1811.08886.pdf)  (CVPR'19, oral)
