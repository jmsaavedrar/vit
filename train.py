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
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import improc.augmentation as aug

import configparser
import argparse
# import the dataset builder, here is an example for qd

#---------------------------------------------------------------------------------------
def map_func(sample, daug_func, n_classes):
    image = sample['image']    
    label = sample['label']
    return daug_func(image), tf.one_hot(label, depth = n_classes)



AUTO = tf.data.AUTOTUNE
#---------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type = str, required = True)    
    parser.add_argument('-model', type = str, choices = ['VIT', 'RESNET'], required = True)
    parser.add_argument('-gpu', type = int, required = False) # gpu = -1 set for using all gpus
    args = parser.parse_args()
    gpu_id = 0
    if not args.gpu is None :
        gpu_id = args.gpu    
    config_file = args.config    
    assert os.path.exists(config_file), '{} does not exist'.format(config_file)            
    #load configuracion file
    config = configparser.ConfigParser()
    config.read(config_file)
    model_name = args.model
    config_model = config[model_name]
    assert not config_model == None, '{} does not exist'.format(model_name)
    
    config_data = config['DATA']
    ds = None
    #
    dataset_name = config_data.get('DATASET')    
    #ds = tfds.load('tfds_qd')
    ds = tfds.load('tfds_skberlin')
        
    #loading dataset example cifar
    daug = aug.DataAugmentation(config_data)
    ds_train = ds['train']
    ds_valid = ds['test'] 
    
    n_valid = len(ds_valid)   
    n_steps_valid = n_valid// config_model.getint('BATCH_SIZE')
    
    ds_train = (
        ds_train.shuffle(1024, seed=config_model.getint('SEED'))
        .map(lambda x: map_func(x,  daug.get_augmentation_fun(), n_classes = config_data.getint('N_CLASSES')), num_parallel_calls=AUTO)
        .batch(config_model.getint('BATCH_SIZE'))
        .prefetch(AUTO) )
    
        
    ds_valid= (
        ds_valid.shuffle(1024, seed=config_model.getint('SEED'))
        .map(lambda x: map_func(x, lambda image:  tf.cast(tf.image.grayscale_to_rgb(image), tf.float32), n_classes = config_data.getint('N_CLASSES')), num_parallel_calls=AUTO)
        .batch(config_model.getint('BATCH_SIZE')))
               
    #----------------------------------------------------------------------------------
    model_dir =  config_model.get('MODEL_DIR')
    if not config_model.get('EXP_CODE') is None :
        model_dir = os.path.join(model_dir, config_model.get('EXP_CODE'))
    model_dir = os.path.join(model_dir, dataset_name, model_name)
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
    
            #model = model.create_vit(config_data, config_model)
            if model_name == 'VIT':
                import models.vit as model 
                model = model.create_vit(config_data, config_model)
                
            if model_name == 'RESNET':
                import models.resnet as model 
                model = model.create_resnet(config_data.getint('N_CLASSES'))
            
            model.compile(optimizer=tf.keras.optimizers.Adam(), #tf.keras.optimizers.SGD(lr_decayed_fn, momentum=0.9),
                           loss= tf.keras.losses.CategoricalCrossentropy(),
                           metrics=['accuracy'])
                        #metrics=['accuracy'tf.keras.metrics.Accuracy()])
            history = model.fit(ds_train,
                                validation_data = ds_valid,
                                validation_steps = n_steps_valid,
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
