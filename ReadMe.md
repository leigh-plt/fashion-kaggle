# Code for [iMaterialist 2019](https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/)

## TPU examples

This code only example how train model (fcn_resnet50 or deeplabv3_resnet50) for image segmentation with TPU and PyTorch

For train use scripts:

``` script/convert_csv ``` - conver csv file to json for faster access

``` script/train.sh ``` - run training loops with TPU

```  script/inference.sh ``` - make submission file

Don't forget : ```conda activate torch-xla-0.5```

For all scripts: TPU_IP_ADDRESS=10.128.17.2 - Replace it on YOUR TPU NODE IP

``` notebook/Fast Visualizing Model.ipynb ``` - contained visualized prediction for trained model on 200 images. Normal training too long.
