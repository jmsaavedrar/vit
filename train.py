"""
This scripts train a ssl model. So far we have

-- SimSiam <adapted from https://keras.io/examples/vision/simsiam/>
-- BYOL

This code use tfds to prepare and load the datasets. Our examepl is with
QuickDraw datasets, so you need the following repository

"""

import sys
import socket
#---------------------------------------------------------------------------
## you can modify this part to set a local path for the dataset project
ip = socket.gethostbyname(socket.gethostname()) 
if ip == '192.168.20.62' :
    sys.path.insert(0,'/home/DIINF/vchang/jsaavedr/Research/git/datasets')
else :
    sys.path.insert(0,'/home/jsaavedr/Research/git/datasets')
#---------------------------------------------------------------------------
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import configparser
import argparse
# import the dataset builder, here is an example for qd

#---------------------------------------------------------------------------------------
def sketch_augment(self, image):
    # As discussed in the SimCLR paper, the series of augmentation
    # transformations (except for random crops) need to be applied
    # randomly to impose translational invariance.        
                
    image = tf.image.grayscale_to_rgb(image)        
    image = self.flip_random_crop(image)
    image = self.random_apply(self.color_jitter, image, p=0.8)
    image = self.random_apply(self.color_drop, image, p=0.2)        
    return image        
#---------------------------------------------------------------------------------------
AUTO = tf.data.AUTOTUNE
#---------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type = str, required = True)    
    parser.add_argument('-gpu', type = int, required = False) # gpu = -1 set for using all gpus
    args = parser.parse_args()
    gpu_id = 0
    if not args.gpu is None :
        gpu_id = args.gpu    
    config_file = args.config
    ssl_model_name = args.model
    assert os.path.exists(config_file), '{} does not exist'.format(config_file)            
    #load configuracion file
    config = configparser.ConfigParser()
    config.read(config_file)
    config_model = config['VIT']
    assert not config_model == None, '{} does not exist'.format(ssl_model_name)
    
    config_data = config['DATA']
    ds = None
    #
    dataset_name = config_data.get('DATASET')    
    ds = tfds.load('tfds_qd')
        
    daug = aug.DataAugmentation(config_data)    
    #loading dataset example cifar
    
    ds_train = ds['train']
    ds_valid = ds['test']    
    ds_train = (
        ds_train.shuffle(1024, seed=config_model.getint('SEED'))
        .map(lambda x: ssl_map_func(x, daug.get_augmentation_fun()), num_parallel_calls=AUTO)
        .batch(config_model.getint('BATCH_SIZE'))
        .prefetch(AUTO) )
    
           
    #----------------------------------------------------------------------------------
    model_dir =  config_model.get('MODEL_DIR')
    if not config_model.get('EXP_CODE') is None :
        model_dir = os.path.join(model_dir, config_model.get('EXP_CODE'))
    model_dir = os.path.join(model_dir, dataset_name, ssl_model_name)
    if not os.path.exists(model_dir) :
        os.makedirs(os.path.join(model_dir, 'ckp'))
        os.makedirs(os.path.join(model_dir, 'model'))
        print('--- {} was created'.format(os.path.dirname(model_dir)))
    #----------------------------------------------------------------------------------
    tf.debugging.set_log_device_placement(True)   
    # Create a cosine decay learning scheduler.
    num_training_samples = len(ds_train)        
    steps = config_model.getint('EPOCHS') * (num_training_samples // config_model.getint('BATCH_SIZE'))
    config_model['STEPS'] = str(steps)  
    # Compile model and start training.
    if gpu_id >= 0 :
        with tf.device('/device:GPU:{}'.format(gpu_id)) :
            lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.03, 
                                                              decay_steps = config_model.getint('STEPS'))
            # Create an early stopping callback.
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", 
                                                      patience=5, 
                                                      restore_best_weights=True)
            
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                                   filepath= os.path.join(model_dir, 'ckp', 'ckp_{epoch:03d}'),
                                                                   save_weights_only=True,                                                                   
                                                                   save_freq = 'epoch',  )
    
            model = vit.vit(config_data, config_model)
            model.compile(optimizer=tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9))
            history = model.fit(ds_train,
                                epochs=config_model.getint('EPOCHS'),
                                callbacks=[early_stopping, model_checkpoint_callback])                
    
                          
    #predicting                    
        #hisitory = simsiam.evaluate(ssl_ds)
    # Visualize the training progress of the model.
    # Extract the backbone ResNet20.
    #saving model
    # print('saving model')
    model_file = os.path.join(model_dir, 'model', 'model')
    
    model.save_weights(model_file)
    print("model saved to {}".format(model_file))        
#