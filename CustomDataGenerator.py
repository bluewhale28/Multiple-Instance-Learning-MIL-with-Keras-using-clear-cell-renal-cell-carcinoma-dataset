import tensorflow as tf
import albumentations as A
import pandas as pd
import numpy as np

class CustomDataGenerator(tf.keras.utils.Sequence):
    """ Create custom DataGenerator
    Parameters
    -------------------
    df (pandas DataFrame) - DataFrame with the data. X (bag of instances) - list of images paths. y -label
    batch_size (int) - size of batch
    input_size (tuple) - size of input image
    shuffle (boolean) - shuffle Dataframe after each epoch. Default - False
    augmentations (boolean) - apply image augmentations state. Augmentations creates with albumentations library. Default - False
    augmentations_list (list) - list of augmentations to perform
    X_col_num (int) - num of X ( list of images paths) column
    y_col_num (int) - num of y (label) column
    Returns
    -------------------
     DataGenerator object
    """
    def __init__(self, df, 
                 batch_size = 1,
                 input_size=(256, 256, 3),
                 shuffle=False,
                 augmentations = False,
                 augmentations_list = [
          A.HorizontalFlip(p=0.2),
          A.RandomRotate90(p=0.2),
          A.Flip(p=0.2),
          A.Transpose(p=0.2),
          ],
                 X_col_num = 1,
                 y_col_num = 2):
        
        self.df = df.copy()
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle        
        self.n = len(self.df)
        self.X_col_num = X_col_num
        self.y_col_num = y_col_num
        self.augmentations_list = augmentations_list
        # Augmentations
        self.augmentations = augmentations
        if self.augmentations == True:
          self.transform = A.Compose(augmentations_list)          
        
    
    def __make_bag(self,images):
       bag = list()                 
       for image in images:
          pic = tf.keras.preprocessing.image.load_img(image)
          pic = tf.keras.preprocessing.image.img_to_array(pic)
          if self.augmentations == True:
            pic = self.transform(image=pic)
            pic = pic["image"]          
          pic = np.expand_dims(pic,0)       
          bag.append(pic)  
       return bag

    
    def on_epoch_end(self):
      if self.shuffle == True:
        self.df = self.df.sample(frac = 1)  
        pass
    
    def __getitem__(self, index):
      images = self.df.iloc[index,self.X_col_num]
      X = self.__make_bag(images)
      y = np.array([self.df.iloc[index,self.y_col_num]])
      y = tf.one_hot(y, depth = 2)

      return X,y
    
    def __len__(self):
        return self.n // self.batch_size
      
    
