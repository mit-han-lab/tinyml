torchpack dist-run -np 16 -H $server1:8,$server2:8 \
python train.py configs/netaug.yaml \
	--data_provider "{data_path:/dataset/imagenet,image_size:144}" \
	--model "{name:tinymbv2}" \
	--netaug "{aug_expand_list:[1.0,1.6,2.2,2.8],aug_width_mult_list:[1.0,1.8,2.6]}" \
	--path <exp_path>
