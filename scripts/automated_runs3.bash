CUDA_VISIBLE_DEVICES=2 python automated_train.py --sim_name=triple_layer --model=unet
CUDA_VISIBLE_DEVICES=2 python automated_train.py --sim_name=straight_waveguide --model=unet
CUDA_VISIBLE_DEVICES=2 python automated_train.py --sim_name=image_sensor --model=unet
CUDA_VISIBLE_DEVICES=2 python automated_train.py --sim_name=triple_layer --model=neurolight