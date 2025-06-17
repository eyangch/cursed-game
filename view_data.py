import models
import tkinter as tk
from PIL import Image, ImageTk
import data
import torch
import torchvision.transforms as transforms

model = models.load_ae("model_ae_16_interp.torch")

window = tk.Tk()
window.title("Cursed QWOP")

label = tk.Label(window)
label.pack()
text_box = tk.Text(window, height=10, width=30)
text_box.pack()

enc_lstm_images, lstm_keys = data.load_np("img_enc.gz"), data.load_np("key_enc.gz")

frame_n = 0

def onKeyPress(event):
    global frame_n
    key = event.char
    if key == 'k':
        frame_n += 1
    elif key == 'j':
        frame_n -= 1
    
    frame_n = max(0, frame_n)

def update_img(i):
    i = frame_n
    model.eval()
    enc_img = torch.tensor(enc_lstm_images[i//256,[i%256]], dtype=torch.float32)
    cur_key = lstm_keys[i//256,[i%256]]
    with torch.no_grad():
        output_img = transforms.ToPILImage()(model.decode(enc_img)[0]).resize((256,256), Image.Resampling.BICUBIC)
        output_tk_img = ImageTk.PhotoImage(output_img)

        label.pil_image = output_img
        label.image = output_tk_img
        label.config(image=output_tk_img)
        label.pack()

        text_box.delete("1.0", "end")
        active_keys = str(i%256) + "\t" + ''.join(["qwop"[i] if cur_key[0,i] == 1 else '' for i in range(4)])
        text_box.insert(tk.END, active_keys)
        text_box.pack()

        if i < 10000:
            window.after(int(20), update_img, i+1)

update_img(0)
window.bind("<KeyPress>", onKeyPress)
window.mainloop()