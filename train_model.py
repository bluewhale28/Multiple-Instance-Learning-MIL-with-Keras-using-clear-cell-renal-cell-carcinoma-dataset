import tensorflow as tf
from CustomDataGenerator import CustomDataGenerator
from SimpleModel import SimpleModel

def train_model (train_df, validation_df, model_save_path):
    """
    Train SimpleModel
    
    Parameters
    -------------------
    train_df (pandas DataFrame) - DataFrame with the training data. X (bag of instances) - list of images paths. y -label
    validation_df (pandas DataFrame) - DataFrame with the validation data. X (bag of instances) - list of images paths. y -label
    model_save_path (str) - path for model saving
    Returns
    -------------------
    """

        
    # create generator of the training and validation data
    train_generator = CustomDataGenerator(df = train_df, shuffle = True, augmentations = True )
    validation_generator = CustomDataGenerator (df = validation_df, shuffle = False, augmentations = False )
    
    # Callbacks
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_save_path,
        monitor="val_loss",
        verbose=1,
        mode="min",
        save_best_only=True,
        save_weights_only= False)
    
    
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1,
        mode="min")
    
    # optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=0.0005, beta_1=0.9, beta_2=0.999)
    
    # create and compile model
    model = SimpleModel(bag_size = 40, instance_shape = (256, 256, 3) )
    model.compile(optimizer = opt, 
    loss='categorical_crossentropy', metrics=["accuracy",tf.keras.metrics.AUC(name = 'AUC'),
                                                        tf.keras.metrics.AUC(curve = 'PR',name = 'PR_AUC'), 
                                                        tf.keras.metrics.Precision(name = 'Precision', class_id = 1),
                                                        tf.keras.metrics.Recall(name = 'Recall',class_id = 1)])
    # model fitting
    model.fit(
        train_generator,
        validation_data = validation_generator ,
        epochs=100,
        batch_size= 1,
        callbacks=[model_checkpoint,es], 
        verbose=1)
