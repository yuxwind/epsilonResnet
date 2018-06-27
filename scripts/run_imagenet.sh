# Example for training and inference on compressed models
#   run epsilon-ResNet 101 with epsilon=2
python imagenetEpsilonResnet.py --gpu 0,1,2,3 --data /scratch/yuxwind/tensorpack_data/ilsvrc12_data/ -d 101 -e 2.0 
#   compress the model of step 550000
mv train_log train_log.imagenet-e_2-d_101
python compressModel.py --dir train_log.imagenet-e_2-d_101 --step 550000
#   do inference on the compressed model of step 550000
python imagenetCompressedResnet.py --cfg train_log.imagenet-e_2-d_101/imagenet-epsilon-resnet/compressed_model_550000.cfg --data /scratch/yuxwind/tensorpack_data/ilsvrc12_data/ --gpu 2  

# Environment:
#   All imangenet training experiments are on DGX machine with 8 Tesla V100-SXM2
# Time:
#   It takes about 86 hours to train epsilon-ResNet 101 with 4 GPUs
#   It takes about 97 hours to train epsilon-ResNet 152 with 6 GPUs
#   
# run epsilon-ResNet 101 with epsilon=2.1
python imagenetEpsilonResnet.py --gpu 0,1,2,3 --data /scratch/yuxwind/tensorpack_data/ilsvrc12_data/ -d 101 -e 2.1 
mv train_log train_log.imagenet-e_2.1-d_101
# run epsilon-ResNet 152 with epsilon=1.8
python imagenetEpsilonResnet.py --gpu 0,1,2,3,4,5 --data /scratch/yuxwind/tensorpack_data/ilsvrc12_data/ -d 152 -e 1.8
mv train_log train_log.imagenet-e_1.8-d_152

# Please change the parameters to compress other models and do inference on it
