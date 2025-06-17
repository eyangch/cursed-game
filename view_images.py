import data
import models
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

imgs = data.load_np("img_dat.gz", "data_full")
model = models.load_ae("model_ae_16_interp.torch")
model.eval()

batch = 0

raw_img = torch.tensor(imgs[batch*256:(batch+1)*256], dtype=torch.float32).swapaxes(1, 3).swapaxes(2, 3) / 256

enc = model.encode(torch.tensor(raw_img)).detach().numpy()

print(enc.shape)

img = model.decode(torch.tensor(enc)).cpu().detach()
lstm_input = torch.unsqueeze(torch.cat((torch.tensor(enc), torch.tensor([[0., 0., 0., 0.]] * 256)), dim=-1), 0)
print(lstm_input.shape)

print(img.shape)

out_imgs = []

model.eval()

def batch_interpretable_loss(encoded, p=2, eps=1e-12):
    encoded_diff = torch.abs(encoded[1:] - encoded[:-1]) / (torch.abs(encoded[:-1])+torch.abs(encoded[1:])+eps)
    normed_diff = F.normalize(encoded_diff, p=p, dim=1)
    batch_losses = torch.sum(normed_diff, dim=1)
    return torch.mean(batch_losses) - 1

for i in range(100):
    pilimg = transforms.ToPILImage()(img[i])
    print(torch.nn.MSELoss()(torch.tensor(enc)[i], torch.tensor(enc)[i+1]))
    dst = pilimg
    #dst = transforms.ToPILImage()(torch.tensor(raw_img[i]))
    out_imgs.append(dst)

print(enc[0].tolist())

torch.set_printoptions(edgeitems=30)
enc_t = torch.tensor(enc)
print(torch.topk(torch.abs(enc_t[1:] - enc_t[:-1]) / (torch.abs(enc_t[:-1])+torch.abs(enc_t[1:])+1e-12), 16, dim=1))

print(batch_interpretable_loss(torch.tensor(enc), p=3) * 10)
out_imgs[0].save("out_test1.gif", save_all=True, append_images=out_imgs[1:], duration=10, loop=0)
