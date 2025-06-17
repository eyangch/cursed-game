import modal
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import models
import time
import numpy as np
import gzip
from config import device

app = modal.App("qwop")
vol = modal.Volume.from_name("qwop", create_if_missing=True)

image = modal.Image.debian_slim().pip_install("torch", "numpy", "pillow").add_local_python_source("config", "models", "player")

def save_np_abs(fname, data):
    with gzip.open(fname, "wb") as f:
        np.save(f, data)

def load_np_abs(fname):
    with gzip.open(fname, "rb") as f:
        return np.load(f)

def convert_data(raw_data):
    return (torch.tensor(raw_data, dtype=torch.float32).swapaxes(1, 3).swapaxes(2, 3) / 256).to(device, memory_format=torch.channels_last)

@app.function(volumes={"/qwop": vol}, image=image, gpu="L4", timeout=900)
def train():
    sample_data = load_np_abs("/qwop/img_dat.gz")
    
    try:
        model = models.load_ae("/qwop/model_ae_16_interp.torch")
        print("loaded model")
    except Exception as e:
        model = models.new_ae()
        print("made new model")
        print(e)

    def loss_fn(recon_x, x):
        return F.mse_loss(recon_x, x)*96*96
    
    def sobel_filter(image, eps):
        image_gs = torch.mean(image, dim=1, keepdim=True)
        blur = torch.tensor([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        image_gs = F.conv2d(image_gs, blur, padding=1)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        img_x = F.conv2d(image_gs, sobel_x, padding=1)
        img_y = F.conv2d(image_gs, sobel_y, padding=1)
        img_res = torch.sqrt(img_x ** 2 + img_y ** 2 + eps)
        return img_res

    def sobel_loss_fn(recon_x, x, eps=1e-6):
        s_recon_x = sobel_filter(recon_x, eps)
        s_x = sobel_filter(x, eps)
        return loss_fn(s_recon_x, s_x)
    
    def batch_interpretable_loss(encoded, p=2, eps=1e-12):
        encoded_diff = torch.abs(encoded[1:] - encoded[:-1]) / (torch.abs(encoded[:-1])+torch.abs(encoded[1:])+eps)
        normed_diff = F.normalize(encoded_diff, p=p, dim=1)
        batch_losses = torch.sum(normed_diff, dim=1)
        return torch.mean(batch_losses) - 1

    def ae_loss_fn(recon_x, x):
        recon_loss = loss_fn(recon_x, x)
        sobel_loss = sobel_loss_fn(recon_x, x)
        return recon_loss + 0.10 * sobel_loss, recon_loss, sobel_loss
    
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 40)

    def generate_data(batch_size, k):

        tensor_data = convert_data(sample_data)

        torch.manual_seed(8765)
        # shuffle_indices = torch.randperm(tensor_data.shape[0]).to(device)
        # tensor_data = tensor_data[shuffle_indices,:,:,:]
        # train_dataloader = DataLoader(tensor_data[:-k*batch_size], batch_size=batch_size, shuffle=True)
        # test_dataloader = DataLoader(tensor_data[-k*batch_size:], batch_size=batch_size, shuffle=True)
        
        num_batches = tensor_data.shape[0] // batch_size
        print(tensor_data.shape)
        batched_data = tensor_data[:num_batches * batch_size].view(num_batches, batch_size, 3, 96, 96)
        torch.manual_seed(8765)
        shuffle_indices = torch.randperm(batched_data.shape[0]).to(device)
        batched_data = batched_data[shuffle_indices,:,:,:,:]
        print(batched_data.shape)
        train_dataloader = DataLoader(batched_data[:-k], batch_size=1, shuffle=True)
        test_dataloader = DataLoader(batched_data[-k:], batch_size=1, shuffle=True)
        return train_dataloader, test_dataloader, tensor_data
    
    def train_ae_epoch(train_dataloader):
        model.train()
        train_loss = 0
        total_interp_loss = 0
        total_rl = 0
        total_cl = 0
        total_train = len(train_dataloader)
        t_train = time.time()
        for idx, images in enumerate(train_dataloader):
            optimizer.zero_grad()
            images = images.squeeze(0)
            encoded_images = model.encode(images)
            recon_images = model(images)
            loss, recon_loss, classify_loss = ae_loss_fn(recon_images, images)
            interp_loss = batch_interpretable_loss(encoded_images, p=2) * 10
            sum_loss = loss + interp_loss
            sum_loss.backward()
            train_loss += loss.item()
            total_interp_loss += interp_loss.item()
            total_rl += recon_loss.item()
            total_cl += classify_loss.item()
            optimizer.step()
            if idx % 10 == 0:
                print(f"[{idx}/{total_train}] - {(idx+1)/(time.time()-t_train)} it/s", end="\r")
        train_loss /= total_train
        total_interp_loss /= total_train
        total_rl /= total_train
        total_cl /= total_train
        model.eval()
        return train_loss, total_rl, total_cl, total_interp_loss

    def test_ae_epoch(test_dataloader):
        model.eval()
        test_loss = 0
        total_interp_loss = 0
        total_rl = 0
        total_cl = 0
        with torch.no_grad():
            for _, images in enumerate(test_dataloader):
                images = images.squeeze(0)
                encoded_images = model.encode(images)
                recon_images = model(images)
                loss, recon_loss, classify_loss = ae_loss_fn(recon_images, images)
                interp_loss = batch_interpretable_loss(encoded_images, p=2) * 10
                test_loss += loss.item()
                total_interp_loss += interp_loss.item()
                total_rl += recon_loss.item()
                total_cl += classify_loss.item()
        test_loss /= len(test_dataloader)
        total_interp_loss /= len(test_dataloader)
        total_rl /= len(test_dataloader)
        total_cl /= len(test_dataloader)
        return test_loss, total_rl, total_cl, total_interp_loss

    def run_ae_epoch(epoch, train_dataloader, test_dataloader):
        t_epoch = time.time()

        train_loss, train_rl, train_cl, train_interp_loss = train_ae_epoch(train_dataloader)
        test_loss, test_rl, test_cl, test_interp_loss = test_ae_epoch(test_dataloader)

        scheduler.step()
        
        print(f"[AE Epoch {epoch}]")
        print(f"\tTrain Loss: {train_loss} [{train_rl}, {train_cl}] | {train_interp_loss}")
        print(f"\tTest Loss:  {test_loss}  [{test_rl}, {test_cl}] | {test_interp_loss}")
        print(f"Finished training epoch in {time.time() - t_epoch}")
        print(f"LR: {scheduler.get_last_lr()}")
        print("-------------------")

        return test_rl
    
    def train(batch_size=256):
        print("TRAINING!!")
        data_generated = 0
        while data_generated < 1:
            data_generated += 1
            print(f"DATA GENERATION {data_generated}:")
            t_start = time.time()
            train_dataloader, test_dataloader, _ = generate_data(batch_size, 30)

            print(f"Got Images in {time.time() - t_start}")

            epoch = 0

            while True:
                epoch += 1

                run_ae_epoch(epoch, train_dataloader, test_dataloader)

                if epoch % 10 == 0:
                    torch.save(model.state_dict(), "/qwop/model_ae_16_interp.torch")
                    vol.commit()

                if epoch > 160:
                    break

    train()


def upload_data(override=False):
    files = [x.path for x in vol.listdir("/")]
    print(files)
    with vol.batch_upload(force=True) as batch:
        if override or "img_dat.gz" not in files:
            batch.put_file("../data_full/img_dat.gz", "/img_dat.gz", )

if __name__ == "__main__":
    upload_data()
    train.remote()