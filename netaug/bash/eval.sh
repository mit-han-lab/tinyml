torchpack dist-run -np 8 python eval.py \
	--dataset imagenet --data_path /dataset/imagenet/ \
	--image_size 160 \
	--model mbv2-0.35 \
	--init_from <path_of_pretrained_weight>

torchpack dist-run -np 8 python eval.py \
	--dataset imagenet --data_path /dataset/imagenet/ \
	--image_size 160 \
	--model mbv3-0.35 \
	--init_from <path_of_pretrained_weight>

torchpack dist-run -np 8 python eval.py \
	--dataset imagenet --data_path /dataset/imagenet/ \
	--image_size 160 \
	--model proxylessnas-0.35 \
	--init_from <path_of_pretrained_weight>

torchpack dist-run -np 8 python eval.py \
	--dataset imagenet --data_path /dataset/imagenet/ \
	--image_size 176 \
	--model mcunet \
	--init_from <path_of_pretrained_weight>

torchpack dist-run -np 8 python eval.py \
	--dataset imagenet --data_path /dataset/imagenet/ \
	--image_size 144 \
	--model tinymbv2 \
	--init_from <path_of_pretrained_weight>
