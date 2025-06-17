import gzip
import numpy as np
import player
import asyncio
import torch
from models import AE
from config import device
import time
from playwright.async_api import async_playwright
from pathlib import Path

def convert_data(raw_data):
    return (torch.tensor(raw_data, dtype=torch.float32).swapaxes(1, 3).swapaxes(2, 3) / 256).to(device, memory_format=torch.channels_last)

def save_np_abs(fname, data):
    with gzip.open(fname, "wb") as f:
        np.save(f, data)

def load_np_abs(fname):
    with gzip.open(fname, "rb") as f:
        return np.load(f)

def save_np(fname, data, folder="data"):
    save_np_abs(f"{folder}/{fname}", data)

def load_np(fname, folder="data"):
    return load_np_abs(f"{folder}/{fname}")
    
async def gen_raw(folder="data", end_time=-1):
    Path(folder).mkdir(parents=True, exist_ok=True)
    try:
        sample_data = load_np("img_dat.gz", folder=folder)
        print(sample_data.shape)
    except:
        sample_data = None

    try:
        sample_keys = load_np("key_dat.gz", folder=folder)
        print(sample_keys.shape)
    except:
        sample_keys = None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        NUM_PAGES = 2
        pages = [await player.setup_page(context) for _ in range(NUM_PAGES)]

        c_t = 0
        while c_t != end_time:
            try:
                gen_data = await asyncio.gather(
                    *[player.play_actions(pages[i], 1000) for i in range(NUM_PAGES)]
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("failed")
                time.sleep(1)
                continue

            c_t += 1
            for data in gen_data:
                new_sample_keys = data[1]
                new_sample_data = data[0]
                new_sample_keys = new_sample_keys

                if sample_data is None:
                    sample_data = new_sample_data
                else:
                    sample_data = np.concatenate((sample_data, new_sample_data))
                
                if sample_keys is None:
                    sample_keys = new_sample_keys
                else:
                    sample_keys = np.concatenate((sample_keys, new_sample_keys))
                
                print(sample_data.shape)
                print(sample_keys.shape)

            if sample_data.shape[0] % 3000 == 0:
                save_np(f"img_{sample_data.shape[0]}.gz", sample_data, folder)
                save_np(f"key_{sample_keys.shape[0]}.gz", sample_keys, folder)

            save_np(f"img_dat.gz", sample_data, folder)
            save_np(f"key_dat.gz", sample_keys, folder)


async def gen_enc(model_ae: AE, folder="data", end_time=-1):
    Path(folder).mkdir(parents=True, exist_ok=True)
    try:
        sample_data = load_np("img_enc.gz", folder)
    except:
        sample_data = None
    
    try:
        sample_keys = load_np("key_enc.gz", folder)
    except:
        sample_keys = None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        NUM_PAGES = 2
        pages = [await player.setup_page(context) for _ in range(NUM_PAGES)]

        c_t = 0
        while c_t != end_time:
            try:
                gen_data = await asyncio.gather(
                    *[player.play_actions(pages[i], 256) for i in range(NUM_PAGES)]
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("failed")
                time.sleep(1)
                continue

            c_t += 1
            for data in gen_data:
                new_sample_keys = data[1]
                new_sample_data = data[0]
                new_sample_keys = np.array([new_sample_keys])

                with torch.no_grad():
                    model_ae.eval()
                    torch_data = convert_data(new_sample_data)
                    enc = np.array([model_ae.encode(torch_data).numpy()])

                if sample_data is None:
                    sample_data = enc
                else:
                    sample_data = np.concatenate((sample_data, enc))
                
                if sample_keys is None:
                    sample_keys = new_sample_keys
                else:
                    sample_keys = np.concatenate((sample_keys, new_sample_keys))
                
                print(sample_data.shape)
                print(sample_keys.shape)

            if sample_data.shape[0] % 2000 == 0:
                save_np(f"img_enc{sample_data.shape[0]}.gz", sample_data, folder)
                save_np(f"key_enc{sample_keys.shape[0]}.gz", sample_keys, folder)

            if sample_data.shape[0] % 250 == 0:
                save_np(f"img_enc.gz", sample_data, folder)
                save_np(f"key_enc.gz", sample_keys, folder)