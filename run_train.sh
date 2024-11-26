#!/bin/bash

echo "Executing: python train.py --model_name 'UnetPlusPlus' --encoder_name 'resnet152' --optimizer adamW --batch_size 4 --resize 512"
python train.py --model_name "UnetPlusPlus" --encoder_name "resnet152" --optimizer adamW --batch_size 4 --resize 512

echo "Executing: python train.py --model_name 'UnetPlusPlus' --encoder_name 'resnet152' --optimizer adamW --batch_size 4 --resize 512 use_clahe True"
python train.py --model_name "UnetPlusPlus" --encoder_name "resnet152" --optimizer adamW --batch_size 4 --resize 512 --use_clahe


echo "Executing: python train.py --model_name 'UnetPlusPlus' --encoder_name 'resnet152' --optimizer adamW --batch_size 1 --resize 1014"
python train.py --model_name "UnetPlusPlus" --encoder_name "resnet152" --optimizer adamW --batch_size 1 --resize 1024

