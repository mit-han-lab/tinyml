torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train.py configs/netaug.yaml \
	--data_provider "{data_path:/dataset/imagenet}" \
	--model "{name:mbv3-0.35}" \
	--run_config "{weight_decay:3.0e-5,base_lr:0.1}" \
	--netaug "{aug_expand_list:[1.0,1.6,2.2,2.8],aug_width_mult_list:[1.0,2.0,3.0]}" \
	--path <exp_path>
