torchpack dist-run -np 32 -H $server1:8,$server2:8,$server3:8,$server4:8 \
python train.py configs/netaug.yaml \
	--data_provider "{data_path:/dataset/imagenet,image_size:176,base_batch_size:64}" \
	--run_config "{base_lr:0.0125}" \
	--model "{name:mcunet}" \
	--netaug "{aug_expand_list:[1.0,1.6,2.2,2.8],aug_width_mult_list:[1.0,1.6,2.2]}" \
	--path <exp_path>
