# mbv2-0.35
torchpack dist-run -np 4 python train.py configs/default.yaml \
	--data_provider "{dataset:food101,base_batch_size:64}" \
	--model "{dropout_rate:0.2}" \
	--run_config "{n_epochs:50,base_lr:0.10}" \
	--init_from <path_of_pretrained_weight> \
	--path <exp_path>

# mcunet
torchpack dist-run -np 4 python train.py configs/default.yaml \
	--data_provider "{dataset:food101,base_batch_size:64,image_size:176}" \
	--model "{dropout_rate:0.1,name:mcunet}" \
	--run_config "{n_epochs:50,base_lr:0.04}" \
	--init_from <path_of_pretrained_weight> \
	--path <exp_path>

# proxylessnas-0.35
torchpack dist-run -np 4 python train.py configs/default.yaml \
	--data_provider "{dataset:food101,base_batch_size:64}" \
	--model "{dropout_rate:0.2,name:proxylessnas-0.35}" \
	--run_config "{n_epochs:50,base_lr:0.10}" \
	--init_from <path_of_pretrained_weight> \
	--path <exp_path>

# mbv3-0.35
torchpack dist-run -np 4 python train.py configs/default.yaml \
	--data_provider "{dataset:food101,base_batch_size:64}" \
	--model "{dropout_rate:0.2,name:mbv3-0.35}" \
	--run_config "{n_epochs:50,base_lr:0.20,weight_decay:3.0e-5}" \
	--init_from <path_of_pretrained_weight> \
	--path <exp_path>

# tinymbv2
torchpack dist-run -np 4 python train.py configs/default.yaml \
	--data_provider "{dataset:food101,base_batch_size:64,image_size:144}" \
	--model "{dropout_rate:0.1,name:tinymbv2}" \
	--run_config "{n_epochs:50,base_lr:0.06}" \
	--init_from <path_of_pretrained_weight> \
	--path <exp_path>
