import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GroupKFold

import numpy as np
import pandas as pd
import os, argparse, gc, datetime
from tqdm import tqdm
from sagemaker_training import environment  # SageMaker training environment helper
                        
def load_dataset(s3_loc, s3_filename, gkf_group_col):
    
    data = np.load(os.path.join(s3_loc, s3_filename))
    feature_cols = [f"feature_{i:02d}" for i in range(79)]
    features = {col: data[col] for col in feature_cols}
    x = np.column_stack([features[col] for col in feature_cols])
    groups = data[gkf_group_col]

    del features, data
    gc.collect()
    
    return x, groups

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.best_weights = None
        self.best_epoch = 1

    def __call__(
        self, 
        test_loss, 
        # model_weights,
        current_epoch,    
    ):
        if test_loss < self.best_loss:
            self.best_loss = test_loss
            self.patience_counter = 0
            # self.best_weights = model_weights
            self.best_epoch = current_epoch
        else:
            self.patience_counter += 1
        
        if self.patience_counter > self.patience:
            return True # Stop training
    
        return False # Continue training

class AutoEncoderDR:
    def __init__(
        self, 
        x_train, x_test, 
        params,
    ):
        self.x_train = x_train
        self.x_test = x_test
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
        
        self.train_ds, self.test_ds = self.create_tf_datasets()
        
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
        
    @tf.function
    def test_step(self, inputs):
        reconstructed = self.model(inputs, training=False)
        loss = self.loss_fn(inputs, reconstructed)
        
        return loss
    
    def create_tf_datasets(self):
        train_dataset = tf.data.Dataset.from_tensor_slices(self.x_train)
        test_dataset = tf.data.Dataset.from_tensor_slices(self.x_test)
        
        train_dataset = (
            train_dataset.shuffle(buffer_size=self.x_train.shape[0])
            .batch(batch_size=self.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        
        test_dataset= (
            test_dataset
            .batch(batch_size=self.batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        
        return train_dataset, test_dataset
    
    def fit(self, early_stopping_callback=None):
        training_loss, test_loss = 0, 0

        for epoch in tqdm(range(self.epochs), leave=False):
            for step, batch in enumerate(self.train_ds, start=1):
                loss = self.train_step(batch)
                training_loss += loss
            training_loss /= step
            
            for step, batch in enumerate(self.test_ds, start=1):
                loss = self.test_step(batch)
                test_loss +=loss
            test_loss /= step
                
            tqdm.write(f"Epoch: {epoch+1}| Training loss: {training_loss:.2f}| Test loss: {test_loss:.2f}")

            if early_stopping_callback is not None:
                if early_stopping_callback(
                    test_loss, 
                    # self.model.weights, 
                    epoch
                ):
                    print(f"Training stopped early in epoch {epoch}")
                    break

    @staticmethod
    def calculate_reconstruction_error(y_true, y_preds):
        return np.mean(np.square(y_true-y_preds))
    
    def evaluate(self, early_stopping_callback=None):
        self.fit(early_stopping_callback=early_stopping_callback)
        reconstructed = self.model(self.x_test)  
        return self.calculate_reconstruction_error(self.x_test, reconstructed) 

class TuningJob:
    
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
        parser.add_argument("--patience", type=int, default=5)
        
        parser.add_argument("--gkf_col", type=str, default="time_id")
        parser.add_argument("--s3_filename", type=str, default="df_530_lite.npz")
        
        # Sagemaker Job parameters
        parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
        parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
        
        return parser.parse_args()

    def apply_crossvalidation(self, x, gkf_groups, patience, params):
        gkf = GroupKFold(n_splits=5)
        cv_error = 0
        cv_early_stopping_epochs = []

        cv_results = {i:j for i, j in params.items()}
        cv_results["fold_errors"] = []
        cv_results["early_stopping_epochs"] = []

        for fold, (train_idx, test_idx) in enumerate(gkf.split(x, x, gkf_groups), start=1):

            with tf.device("/device:GPU:0"):
                ae = AutoEncoderDR(
                    x[train_idx], x[test_idx], 
                    params
                )

                # Reset early stopping callback state every fold
                early_stopping = EarlyStopping(patience=patience)
                fold_error = ae.evaluate(early_stopping_callback=early_stopping)
                
            cv_error += fold_error

            cv_results["fold_errors"].append(np.round(fold_error, 4))
            cv_results["early_stopping_epochs"].append(early_stopping.best_epoch)

        cv_results["mean_cv_error"] = np.round(cv_error/fold, 4) 

        with open(os.path.join(environment.output_dir, "metrics.json"), "w") as f:
            f.write(f'{{"mean_cv_error": {(cv_error/fold):.4f}}}')

        return cv_results
            
    def save_cv_results(self):
        current_ts = datetime.datetime.strftime(datetime.datetime.now(), format="%m%d_%H%M%s")
        pd.DataFrame([self.cv_results], index=[0]).to_csv(os.path.join(environment.output_dir, f"cv_{current_ts}.csv"))
        
    def run(self):
        args = self.parse_args()

        x, gkf_groups = load_dataset(args.train, args.s3_filename, args.gkf_col)
        params = {
            "n_enc_1": args.n_enc_1,
            "n_enc_2": args.n_enc_2,
            "n_dec_1": args.n_dec_1,
            "n_dec_2": args.n_dec_2,
            "epochs": args.epochs,
            "lr": args.learning_rate,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
        }
        self.cv_results = self.apply_crossvalidation(x, gkf_groups, args.patience, params)
        self.save_cv_results()
        
if __name__ == "__main__":
    tj = TuningJob()
    tj.run()