import torch
from torch import nn
import torch.nn.functional as F
import math
# from load_data_b import CustomDataLoader, DummyDataLoader
import torchaudio
import os
import glob
import numpy as np
from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from timeit import default_timer as timer
# from flash_attention_mha import MHA
import math
# from flash_attention import FlashTransformerEncoder
import clip

is_cuda = torch.cuda.is_available()
device = torch.device("cuda:0") if is_cuda else torch.device("mps:0")

model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(["a dog eating a cake"]).to(device)
text_features = model.encode_text(text)
print(text_features)
exit()



# x = torch.tensor([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# batch_size, h, w, channels = x.shape
# window_size = 4
# x = x.unfold(1, window_size, window_size).unfold(2, window_size, window_size).contiguous().view(-1, window_size*window_size, self.layer_1_d_model)
# # x = self.transformer_1(x)
# # x = self.layer_1_downsample(x)
# x = x.reshape(batch_size, h//4, w//4, -1)

# x = x.unfold(1, window_size, window_size).unfold(2, window_size, window_size).contiguous().view(-1, window_size*window_size, 4*16)

# print(x)
# exit()

# # x = self.transformer_1(x)
# # x = self.layer_1_downsample(x)
# # x = self.layer_2(x)



model_name = "net_1"
for folder in [f"{model_name}_out", f"{model_name}_checkpoints"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

factor = 4
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        size = 64
        layers_count = int(math.log(size, 4))

        self.layer_1_d_model = layer_1_d_model = 8
        self.transformer_1 = FlashTransformerEncoder(6, layer_1_d_model, 2, layer_1_d_model * 4, dropout=0.01)
        self.layer_1_downsample = nn.Linear(layer_1_d_model, 4)

        self.layer_2_d_model = layer_2_d_model = 4*16
        self.transformer_2 = FlashTransformerEncoder(6, layer_2_d_model, 8, layer_2_d_model * 4, dropout=0.01)

        self.layer_3_d_model = layer_3_d_model = layer_2_d_model * 4
        self.transformer_2 = FlashTransformerEncoder(15, layer_3_d_model, 16, layer_3_d_model * 4, dropout=0.01)

    def forward(self, img, text_features):
        # shape is batch_size, h, w, channels
        batch_size, h, w, channels = img.shape
        window_size = 4
        img = img.unfold(1, window_size, window_size).unfold(2, window_size, window_size).contiguous().view(-1, window_size*window_size, self.layer_1_d_model)
        img = self.transformer_1(img)
        img = self.layer_1_downsample(img)
        img = img.reshape(batch_size, h//4, w//4, -1)

        img = img.unfold(1, window_size, window_size).unfold(2, window_size, window_size).contiguous().view(-1, window_size*window_size, self.layer_2_d_model)
        img = self.transformer_2(img)
        img = img.reshape(batch_size, h//16, w//16, self.layer_2_d_model)


        img = img.unfold(1, window_size, window_size).unfold(2, window_size, window_size).contiguous().view(-1, window_size*window_size, self.layer_3_d_model)
        img = self.transformer_3(img)

        # img = img.reshape(batch_size, -1, self.layer_3_d_model)


        img = self.transformer_1(img)
        img = self.layer_1_downsample(img)
        img = self.layer_2(img)
        return img
    
def format_data(x):
    # x is PIL image
    return x



if __name__ == "__main__":
    writer = SummaryWriter(log_dir=f"runs/{model_name}")
    for f in glob.glob(f'{model_name}_out/*'):
        os.remove(f)
    segment_length = 2**15 # = 32768 approx 1s
    model = Net(segment_length, device=device).to(device)
    # model=torch.compile(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("num trainable params is", params)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=.0001)
    criterion = torch.nn.MSELoss()
    sample_rate = 32000
    step=0
    total_steps=-1
    start_epoch = 0

    checkpoints_sorted = glob.glob(f'{model_name}_checkpoints/*.pt')
    if len(checkpoints_sorted) > 0:
        checkpoints_sorted.sort(key=os.path.getmtime)
        print("loading checkpoint", checkpoints_sorted[-1],)
        checkpoint = torch.load(checkpoints_sorted[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        total_steps = checkpoint['total_steps'] if ('total_steps' in checkpoint) else 8999
        step = checkpoint['step']
        start_epoch = checkpoint['epoch']
        print("loaded checkpoint", checkpoints_sorted[-1], "total steps", total_steps, "start_i", step, "epoch", start_epoch)
    
    target_lr=.000001
    lr = target_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    warmup_steps = 1000
    for epoch in range(start_epoch, 160):
        dataset = CustomDataLoader(250, 12, sample_rate, segment_length=segment_length, max_segments_per_file=20, start_index=step)
        # dataset = DummyDataLoader(segment_length)
        loop_time = timer()
        for i, (x, target) in enumerate(dataset, step):
            # print("time loading", timer() - loop_time);
            start = timer()
            x = torch.from_numpy(x.copy()).unsqueeze(0).to(device)
            target = torch.from_numpy(target.copy()).unsqueeze(0).to(device)
            end = timer()
            step += 1
            total_steps += 1
            if total_steps < warmup_steps:
                lr = (target_lr / warmup_steps) * (total_steps+1)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            out_all = []
            batch_size = 8
            # for j in range(0, model.sub_segments, batch_size):
                # print(j)
                # target = model.get_subsegment(target_all, j, batch_size)
                    
            optimizer.zero_grad()

            if is_cuda:
                with autocast(dtype=torch.bfloat16):
                    out = model(x)
                    loss = criterion(out, target)
            else:
                out = model(x)
                loss = criterion(out, target)

            if out.isinf().any() or out.isnan().any():
                print("out is inf or nan")
                continue

            loss_item = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if total_steps % 10 == 0:
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                    break
                print(out, out.max().item(), out.min().item())
                print(out[0,0,253:257])
                print(f"epoch {epoch}, step: {step}, loss: {str(loss.item())[:8]}, lr: {str(lr)} min {str(torch.min(out).item())}, max {str(torch.max(out).item())}, avg {str(torch.mean(out.abs()).item())}, t_min {str(torch.min(target).item())[:8]}, t_max {str(torch.max(target).item())[:8]}, t_avg {str(torch.mean(target.abs()).item())[:8]}")
                writer.add_scalar("lr", lr, total_steps)
                writer.add_scalar("main_loss", loss_item, total_steps)
                writer.add_scalar("min", torch.min(target).item(), total_steps)
                writer.add_scalar("max", torch.max(target).item(), total_steps)
                writer.flush()

            optimizer.step()

            if total_steps % 100 == 0:
                torchaudio.save(f"{model_name}_out/e_{str(epoch)}_i_{str(i)}_out.wav", out.detach().reshape(1,-1).cpu().float(), sample_rate)
                torchaudio.save(f"{model_name}_out/e_{str(epoch)}_i_{str(i)}_in.wav", x.detach().reshape(1,-1).cpu().float(), sample_rate)
                torchaudio.save(f"{model_name}_out/e_{str(epoch)}_i_{str(i)}_target.wav", target.detach().reshape(1,-1).cpu().float(), sample_rate)

            if total_steps % 1000 == 0 and step != 0:
                old_checkpoints = glob.glob(f"{model_name}_checkpoints/*")
                old_checkpoints.sort(key=os.path.getmtime)
                for f in old_checkpoints[:-1]:
                    os.remove(f)
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'total_steps': total_steps,
                    'loss': loss
                }, f"{model_name}_checkpoints/{str(epoch)}_{str(i)}.pt")
            loop_time = timer()
        step = 0