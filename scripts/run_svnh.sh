# run epsilon ResNet 110 with epsilon=1.5
python svhnEpsilonResnet.py --gpu 2 -n 18 -e 1.5 -o svhn-e_1.5-n_18
#todo
# compress the model of step 303420
python compressModel.py --dir train_log.svhn-e_1.5-n_18 --step 303420
# inference on the compressed model of step 303420
python svhnCompressedResnet.py --cfg train_log.svhn-e_1.5-n_18/svhnEpsilonResnet.py/compressed_model_303420.cfg --gpu 2 


# run epsilon ResNet 200 with epsilon=1.5
python svhnEpsilonResnet.py --gpu 2 -n 33 -e 1.5 -o svhn-e_1.5-n_33
# run epsilon ResNet 500 with epsilon=1.5
python svhnEpsilonResnet.py --gpu 2 -n 50 -e 1.5 -o svhn-e_1.5-n_50

# Please change the parameter to compress other model and do inference
