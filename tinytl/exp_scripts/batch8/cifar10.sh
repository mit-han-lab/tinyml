python tinytl_fgvc_train.py --transfer_learning_method tinytl-lite_residual+bias \
    --train_batch_size 8 --test_batch_size 100 \
    --n_epochs 50 --init_lr 3e-4 --opt_type adam \
    --label_smoothing 0.1 --distort_color None --frozen_param_bits 8 \
    --gpu 0 --dataset cifar10 --path .exp/batch8/cifar10