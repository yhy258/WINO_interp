CUDA_VISIBLE_DEVICES=0 python automated_train.py --sim_name=triple_layer --model=wino
CUDA_VISIBLE_DEVICES=0 python automated_train.py --sim_name=straight_waveguide --model=wino
CUDA_VISIBLE_DEVICES=0 python automated_train.py --sim_name=image_sensor --model=wino


CUDA_VISIBLE_DEVICES=1 python automated_train.py --sim_name=triple_layer --model=fno2d
CUDA_VISIBLE_DEVICES=1 python automated_train.py --sim_name=straight_waveguide --model=fno2d
CUDA_VISIBLE_DEVICES=1 python automated_train.py --sim_name=image_sensor --model=fno2d
CUDA_VISIBLE_DEVICES=1 python automated_train.py --sim_name=straight_waveguide --model=neurolight


CUDA_VISIBLE_DEVICES=2 python automated_train.py --sim_name=triple_layer --model=unet
CUDA_VISIBLE_DEVICES=2 python automated_train.py --sim_name=straight_waveguide --model=unet
CUDA_VISIBLE_DEVICES=2 python automated_train.py --sim_name=image_sensor --model=unet
CUDA_VISIBLE_DEVICES=2 python automated_train.py --sim_name=triple_layer --model=neurolight

CUDA_VISIBLE_DEVICES=3 python automated_train.py --sim_name=triple_layer --model=fno2dfactor
CUDA_VISIBLE_DEVICES=3 python automated_train.py --sim_name=straight_waveguide --model=fno2dfactor
CUDA_VISIBLE_DEVICES=3 python automated_train.py --sim_name=image_sensor --model=fno2dfactor
CUDA_VISIBLE_DEVICES=3 python automated_train.py --sim_name=image_sensor --model=neurolight
