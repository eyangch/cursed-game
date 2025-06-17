from PIL import Image, ImageTk
import torch
import models
import torchvision.transforms as transforms
import tkinter as tk
from config import lat

model = models.load_ae("model_ae_16_interp.torch")    
model_lstm = models.load_lstm("model_lstm_16_interp.torch")

model.eval()
model_lstm.eval()

model_lstm.set_batch_size(1)
model_lstm.zero_hidden()

window = tk.Tk()
window.title("Cursed QWOP")

label = tk.Label(window)
label.pack()

key_inputs = [0, 0, 0, 0]

reset_state = torch.tensor([[-0.5014382004737854, 0.745929479598999, 1.004421353340149, -1.0390024185180664, -0.3761645555496216, -0.6434043645858765, 4.036914825439453, 0.7974051237106323, -3.7382349967956543, -0.26134952902793884, 1.162588357925415, 0.7374235987663269, 0.6095333695411682, 1.9149888753890991, -0.6184295415878296, -0.449447333812713,
          0.0000,  0.0000,  0.0000,  0.0000]])

mu_diff = torch.tensor([-6.8501e-05, -6.7174e-05,  7.8332e-05, -2.3016e-04,  1.2659e-06,
         3.4621e-09,  6.1905e-03,  6.9170e-04, -5.9594e-03, -1.4974e-03,
         3.6950e-04, -3.9456e-04,  9.0857e-09,  2.1635e-03, -8.4799e-09,
         8.8101e-05])

sigma_diff = torch.tensor([1.6053e-01, 3.0481e-01, 6.5937e-02, 7.8437e-02, 8.9843e-04, 1.2903e-06,
        1.4180e+00, 5.0182e-01, 1.4436e+00, 4.9746e-01, 3.5149e-01, 8.5427e-01,
        2.6917e-06, 3.4817e-01, 2.3050e-06, 2.7934e-01])

def onKeyPress(event):
    global key_inputs
    global state
    global model_lstm

    key = event.char
    if key == 'r':
        state = reset_state
        model_lstm.zero_hidden()
    key_dict = {
        'q': 0,
        'w': 1,
        'o': 2,
        'p': 3,
        'z': 4
    }
    if key in key_dict:
        key_inputs[key_dict[key]] = 1

def onKeyRelease(event):
    global key_inputs
    key_dict = {
        24: 0,
        25: 1,
        32: 2,
        33: 3,
        'z': 4
    }
    if event.keycode in key_dict:
        key_inputs[key_dict[event.keycode]] = 0

state = reset_state
def update_img(i):
    global state
    model_lstm.eval()
    model.eval()
    with torch.no_grad():
        output_img = transforms.ToPILImage()(model.decode(state[[0],:lat])[0]).resize((256,256), Image.Resampling.BICUBIC)
        output_tk_img = ImageTk.PhotoImage(output_img)
        state = torch.unsqueeze(state, 0)
        lstm_output = (mu_diff + model_lstm(state) * sigma_diff)[0] + state[0,:,:lat]

        label.pil_image = output_img
        label.image = output_tk_img
        label.config(image=output_tk_img)
        label.pack()

        new_state = torch.cat((lstm_output, torch.tensor([key_inputs], dtype=torch.float32)), dim=-1)
        state = new_state

        if i < 10000:
            window.after(int(35), update_img, i+1)

update_img(0)
window.bind("<KeyPress>", onKeyPress)
window.bind("<KeyRelease>", onKeyRelease)
window.mainloop()