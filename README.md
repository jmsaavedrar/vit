# vit
This project includes implementations of ResNet and ViT. In addition, it includes a ResNet + Attention (a simple hybrid model).

# Datasets
In this implementation, we use datasets in TFDS format. As an example, we have a dataset of sketches (from TUI-BERLIN) stored in TFDS. You can download it from [here](https://www.dropbox.com/scl/fi/2solxdfth188dug5bjqia/tfds_skberlin.tar?rlkey=2m0mhnvzugb86ovafwrr09kld&dl=0). You can view the images in the dataset using https://github.com/jmsaavedrar/datasets/blob/main/view_dataset.py through the following command:

$ python view_dataset.py  -dataset tfds_skberlin -data test


You will need to untar the dataset into $(HOME)/tensorflow_datasets  (this is the folder (by default) where the system looks for tfds datasets).

# Train
$ python train.py  -config config/berlin.ini -model <MODEL> -gpu 0

where MODEL  should be replaced by RESNET, RESNET-ATT or VIT

# Evaluate
$ python train.py  -config config/berlin.ini -model <MODEL> -gpu 0 -test_only
