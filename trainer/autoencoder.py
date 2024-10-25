import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
import os, gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import pandas as pd
import argparse, datetime
from tqdm import tqdm

def load_dataset(s3_loc, s3_filename):
    
    data = np.load(os.path.join(s3_loc, s3_filename))
    feature_cols = [f"feature_{i:02d}" for i in range(79)]
    features = {col: data[col] for col in feature_cols}
    x = np.column_stack([features[col] for col in feature_cols])
    print(f"Input shape: {x.shape}")
    del features
    gc.collect()
    
    return x

class AutoEncoderTrainOnly:
    def __init__(self, x_train, params):      
        self.x_train = x_train
        self.input_size = x_train.shape[1]
        self.latent_dim = params["latent_dim"]
        
        self.n_enc_1 = params["n_enc_1"]
        self.n_enc_2 = params["n_enc_2"]
        self.n_dec_1 = params["n_dec_1"]
        self.n_dec_2 = params["n_dec_2"]
        
        self.lr = params["lr"]
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        
        self.model = self.build_autoencoder()
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
        self.train_ds = self.create_tf_datasets()
        
        self.model_dir = params["model_dir"]
        
    def build_autoencoder(self):
        
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        
        input_layer = tf.keras.Input(shape=(self.input_size,))
        encoded_output = self.encoder(input_layer)
        decoded_output = self.decoder(encoded_output)
        model = Model(inputs=input_layer, outputs=decoded_output)
        # model.compile(optimizer=self.optimizer, loss=self.loss_fn,)
        
        return model
        
    def build_encoder(self):
        encoder = Sequential([
            Dense(self.n_enc_1, activation="relu", input_shape=(self.input_size,)),
            Dense(self.n_enc_2, activation="relu"),
            Dense(self.latent_dim, activation="relu")
        ])
        
        return encoder
        
    def build_decoder(self):
        decoder= Sequential([
            Dense(self.n_dec_1, activation="relu", input_shape=(self.latent_dim, )),
            Dense(self.n_dec_2, activation="relu"),
            Dense(self.input_size, activation=None)
        ])
        return decoder
        
    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            reconstructed = self.model(inputs, training=True)
            loss = self.loss_fn(inputs, reconstructed)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        return loss
                  
    def create_tf_datasets(self):
        train_dataset = tf.data.Dataset.from_tensor_slices(self.x_train)
        
        train_dataset = (
            train_dataset.shuffle(buffer_size=100000)
            .batch(batch_size=self.batch_size)
            # .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            .prefetch(buffer_size=2) # prefetch next 2 batches in memory while training
        )
        
        return train_dataset
    
    def fit(self):
        for epoch in tqdm(range(self.epochs), leave=False):
            training_loss = 0
            for step, batch in enumerate(self.train_ds, start=1):
                loss = self.train_step(batch)
                training_loss += loss
            training_loss /= step
            
            tqdm.write(f"Epoch: {epoch+1}| Training loss: {training_loss:.2f}")
    
    def save_artifacts(self, save_weights=True):
         #---
        # Save model
        ckpt_dir = self.model_dir
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
        
        if save_weights:
            weights_suffix = ".weights"
        else: 
            weights_suffix = ""
        
        model_save_path = os.path.join(ckpt_dir, f"model{weights_suffix}.h5")
        encoder_save_path = os.path.join(ckpt_dir, f"encoder{weights_suffix}.h5")
        decoder_save_path = os.path.join(ckpt_dir, f"decoder{weights_suffix}.h5")
        
        if save_weights:
            self.model.save_weights(model_save_path)
            self.encoder.save_weights(encoder_save_path)
            self.decoder.save_weights(decoder_save_path)
        else:
            self.model.save(model_save_path)
            self.encoder.save(encoder_save_path)
            self.decoder.save(decoder_save_path)
        #---
        # Save encoded output
        x_encoded = self.encoder.predict(self.x_train)
        encoded_save_path = os.path.join(ckpt_dir, "x_encoded.npz")
        np.savez_compressed(encoded_save_path, array=x_encoded)
        #---
    
class TrainingJob:
    
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()
        
        # Hyperparameters
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--latent_dim", type=int, default=16)
        parser.add_argument("--n_enc_1", type=int, default=64)
        parser.add_argument("--n_enc_2", type=int, default=64)
        parser.add_argument("--n_dec_1", type=int, default=64)
        parser.add_argument("--n_dec_2", type=int, default=64)
        
        parser.add_argument("--s3_filename", type=str, default="df_530_lite.npz")
        
        # Sagemaker Job parameters
        parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
        parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
        
        return parser.parse_args()
        
    def run(self):
        args = self.parse_args()
        x = load_dataset(args.train, args.s3_filename)
        params = {
            "n_enc_1": args.n_enc_1,
            "n_enc_2": args.n_enc_2,
            "n_dec_1": args.n_dec_1,
            "n_dec_2": args.n_dec_2,
            "epochs": args.epochs,
            "lr": args.learning_rate,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "model_dir": args.model_dir,
        }
        
        with tf.device("/device:GPU:0"):
            ae = AutoEncoderTrainOnly(x, params)
            ae.fit()
            ae.save_artifacts()
        
if __name__ == "__main__":
    tj = TrainingJob()
    tj.run()
    
    
