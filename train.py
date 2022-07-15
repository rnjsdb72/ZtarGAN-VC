from tqdm import tqdm
import torch

def preprocess(network, samples, device):
    samples = samples.to(device)
    mel_spect = network.mel_extractor(samples)
    if mel_spect.shape[-1] % 2 == 0:
        mel_spect = mel_spect[...,:-1] # discard the last frame to make divisible by 2
    x,y = mel_spect[...,:-1], mel_spect[...,1:]
    noise = torch.normal(mean=torch.zeros(x.shape[0],1,1)).to(device)
    return x.transpose(1,2).contiguous(), y.transpose(1,2).contiguous(), noise  # transpose so its (batch_sz, time, mels)

def train(network, optimizer, tr_data, va_data, config, device, path="model_checkpoints"):
    train_loss_list = []
    valid_loss_list = []
    best_loss = np.inf
    for epoch in range(config.epochs):
        network.train() # set to training mode
        for samples,cond in tqdm(tr_data):
            x,y,noise = preprocess(network, samples, device)
            pred_params = network(x.requires_grad_(), cond.to(device), noise)
            batch_loss = loss_fn(*pred_params, y)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss_list.append(float(batch_loss.item()))
        va_loss = 0
        network.eval()  # set to evaluation mode
        for i, (samples,cond) in enumerate(va_data):
            x,y,noise = preprocess(network, samples, device)
            pred_params = network(x, cond.to(device), noise)
            batch_loss = loss_fn(*pred_params, y)
            va_loss += float(batch_loss.item())
        del pred_params, batch_loss  # memory leak? (it actually appears to be)
        valid_loss_list.append(va_loss)
        if va_loss < best_loss:# and epoch - last_save > 5: # only save max every 5 epochs
            print("Saving network at epoch", epoch)
            network.save(epoch, va_loss, optimizer, path=path)
            best_loss = va_loss
        print(va_loss)