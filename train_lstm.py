import time
import torch
from torch import optim
import random
import models
from models import GameLSTM
import gzip
import numpy as np
from config import device, lat
import modal

app = modal.App("qwop")
vol = modal.Volume.from_name("qwop", create_if_missing=True)

image = modal.Image.debian_slim().pip_install("torch", "numpy", "pillow").add_local_python_source("config", "models")

def save_np_abs(fname, data):
    with gzip.open(fname, "wb") as f:
        np.save(f, data)

def load_np_abs(fname):
    with gzip.open(fname, "rb") as f:
        return np.load(f)
    
def weighted_mse(output, label, weights):
    return torch.mean((output - label) ** 2 * weights)

def unweighted_mse(output, label, weights):
    return torch.mean((output - label) ** 2)

@app.function(volumes={"/qwop": vol}, image=image, gpu="L4", timeout=900)
def train_lstm(batch_size=32, k=100):
    try:
        model_lstm = models.load_lstm("/qwop/model_lstm_16_interp.torch").to(device)
    except:
        model_lstm = GameLSTM().to(device)

    model_lstm.set_batch_size(batch_size)
    optimizer_lstm = optim.AdamW(model_lstm.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_lstm, 20)
    print("TRAINING LSTM!!")
    data_generated = 0
    while True:
        data_generated += 1
        print(f"DATA GENERATION {data_generated}:")
        t_start = time.time()
        enc_lstm_images, lstm_keys = load_np_abs("/qwop/img_enc.gz"), load_np_abs("/qwop/key_enc.gz")

        diffs = np.reshape(enc_lstm_images[:,:-1] - enc_lstm_images[:,1:], (-1, lat))
        mu_diff = np.mean(diffs, axis=0)
        sigma_diff = np.sqrt(np.mean((diffs - mu_diff) ** 2, axis=0))
        mu_diff = torch.tensor(mu_diff).to(device)
        sigma_diff = torch.tensor(sigma_diff).to(device)
        print(mu_diff)
        print(sigma_diff)
        print(np.max(diffs, axis=0))

        encoded_img_tensor = torch.tensor(enc_lstm_images, dtype=torch.float32).to(device)
        lstm_key_tensor = torch.tensor(lstm_keys, dtype=torch.float32).to(device)

        encoded_img_output = encoded_img_tensor[:,1:,:]
        encoded_img_input = torch.cat((encoded_img_tensor, lstm_key_tensor), dim=-1)[:,:-1,:]

        print(encoded_img_input.shape)
        print(encoded_img_output.shape)

        batch_in = list(torch.split(encoded_img_input, batch_size))[:-1] # :-1 to get rid of incomplete last batch
        batch_out = list(torch.split(encoded_img_output, batch_size))[:-1]

        batches = list(zip(batch_in, batch_out))
        random.seed(6789)
        random.shuffle(batches)

        batch_train = batches[:-k]

        batch_test = batches[-k:]

        print(len(batch_train))
        print(len(batch_test))

        print(f"Got Images in {time.time() - t_start}")

        epoch = 0

        teach_rate = 1.0

        while True:
            t_epoch = time.time()
            train_loss = 0
            train_loss_base = 0
            for i, (image, label) in enumerate(batch_train):
                model_lstm.zero_hidden()

                mult = torch.randn(image.shape).to(device) * 0.10 * torch.cat((sigma_diff, torch.zeros(4).to(device))).to(device)
                noised_image = image + mult

                if teach_rate == 1:
                    lstm_output = mu_diff + model_lstm(noised_image) * sigma_diff + noised_image[:,:,:lat]

                base_output = noised_image[:,:,:lat]
                loss = weighted_mse(lstm_output, label, 1/sigma_diff)
                base_loss = weighted_mse(base_output, label, 1/sigma_diff)
                
                loss = loss / (base_loss + 1e-12)

                optimizer_lstm.zero_grad()
                loss.backward()
                train_loss += loss.item()
                train_loss_base += base_loss.item()
                optimizer_lstm.step()
                scheduler.step(epoch + i/len(batch_train))
            
            epoch += 1
            train_loss /= len(batch_train)
            train_loss_base /= len(batch_train)

            test_loss = 0
            test_loss_base = 0
            with torch.no_grad():
                for image, label in batch_test:
                    model_lstm.zero_hidden()
                    noised_image = image
                    lstm_output = mu_diff + model_lstm(noised_image) * sigma_diff + noised_image[:,:,:lat]
                    base_output = noised_image[:,:,:lat]
                    loss = weighted_mse(lstm_output, label, 1/sigma_diff)
                    base_loss = weighted_mse(base_output, label, 1/sigma_diff)
                    loss = loss / (base_loss + 1e-12)
                    test_loss += loss.item()
                    test_loss_base += base_loss.item()
            
            test_loss /= len(batch_test)
            test_loss_base /= len(batch_test)

            
            print(f"[Epoch {epoch}]")
            print(f"\tTrain Loss: {train_loss} ({train_loss_base})")
            print(f"\tTest Loss:  {test_loss} ({test_loss_base})")
            print(f"Finished training epoch in {time.time() - t_epoch}")
            print(f"Scheduler LR: {scheduler.get_last_lr()}")
            print("-------------------")
            
            if test_loss < 0.0:
                teach_rate *= 0.9
                print(f"NEW TEACH RATE: {teach_rate}")

            if epoch % 2 == 0:
                print("SAVING")
                torch.save(model_lstm.state_dict(), "/qwop/model_lstm_16_interp.torch")

            if epoch > 10000:
                break

def upload_data(override=False):
    files = [x.path for x in vol.listdir("/")]
    print(files)
    with vol.batch_upload(force=True) as batch:
        if "img_enc.gz" not in files or override:
            batch.put_file("data/img_enc.gz", "/img_enc.gz")
        if "key_enc.gz" not in files or override:
            batch.put_file("data/key_enc.gz", "/key_enc.gz")

@app.local_entrypoint()
def main():
    print("Uploading")
    upload_data(False)
    print("Finished Uploading")
    train_lstm.remote()

if __name__ == "__main__":
    print("running locally")
    train_lstm.local()