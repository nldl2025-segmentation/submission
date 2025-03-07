# prototyping of 3D probabilistic network with skip connections

import torch.nn as nn
import torch.utils
import torch.distributions as dist
import numpy as np
from torch.distributions.kl import kl_divergence as kl_div
import nibabel as nib
import os
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import argparse
import gc



class simple_VAE(nn.Module):
    def __init__(self, image_size,  device, dims, latent_size, kernel_size = 5, maxpool_stride = 1, conv_stride = 1, predictive = False):
        super(simple_VAE, self).__init__()
        self.image_size = torch.prod(torch.tensor(image_size)).item()
        self.predictive = predictive
        if image_size[-1] != image_size[-2]:
            print(image_size)
            raise ValueError('Only square images are supported')
        

        dec_modules = []
        enc_modules = []
        enc_sigmas = []
        #dec_sigmas = []
        enc_mus = []
        dec_linear = []



        tconv_padding = (kernel_size-maxpool_stride)//2
        output_padding = 0

        self.sizes = [image_size[-1]//(maxpool_stride**(i)) for i in range(len(dims)-1)]
        linear_size = self.sizes[-1]
        
        checks = [self.sizes[i]%maxpool_stride for i in range(len(dims)-1)]
        print(self.sizes)


        if sum(checks) != 0:
            print("Sizes: ", self.sizes)
            print("Checks: ", checks)
            raise ValueError('Only even dimensions are supported')


        
        if maxpool_stride >2:
            raise ValueError('Only maxpool strides of 1 or 2 are supported')
        #self.sizes = self.sizes[::-1]
        #linear_size = linear_size//maxpool_stride

        self.enc1 = nn.Sequential(nn.Conv3d(in_channels = dims[0], out_channels = dims[1], kernel_size = kernel_size, stride = conv_stride, padding = kernel_size-2),
                                    nn.BatchNorm3d(dims[1]),nn.ReLU(),
                                    nn.Conv3d(in_channels = dims[1], out_channels = dims[1], kernel_size = kernel_size, stride = conv_stride, padding = kernel_size-2),
                                    nn.BatchNorm3d(dims[1]),nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv3d(in_channels = dims[1], out_channels = dims[2], kernel_size = kernel_size, stride = conv_stride, padding = kernel_size-2),
                                    nn.BatchNorm3d(dims[2]),nn.ReLU(),
                                    nn.Conv3d(in_channels = dims[2], out_channels = dims[2], kernel_size = kernel_size, stride = conv_stride, padding = kernel_size-2),
                                    nn.BatchNorm3d(dims[2]),nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv3d(in_channels = dims[2], out_channels = dims[3], kernel_size = kernel_size, stride = conv_stride, padding = kernel_size-2),
                                    nn.BatchNorm3d(dims[3]),nn.ReLU(),
                                    nn.Conv3d(in_channels = dims[3], out_channels = dims[3], kernel_size = kernel_size, stride = conv_stride, padding = kernel_size-2),
                                    nn.BatchNorm3d(dims[3]),nn.ReLU())
        self.enc_mu_1 = nn.Linear(self.sizes[0], latent_size)
        self.enc_sigma_1 = nn.Linear(self.sizes[0], latent_size)
        self.enc_mu_2 = nn.Linear(self.sizes[1], latent_size)
        self.enc_sigma_2 = nn.Linear(self.sizes[1], latent_size)
        self.enc_mu_3 = nn.Linear(self.sizes[2], latent_size)
        self.enc_sigma_3 = nn.Linear(self.sizes[2], latent_size)
        # self.enc_mu_4 = nn.Linear(self.sizes[4], latent_size)
        # self.enc_sigma_4 = nn.Linear(self.sizes[4], latent_size)
        self.maxpool = nn.MaxPool3d(kernel_size=kernel_size, stride = maxpool_stride, padding = kernel_size-2)

        self.sizes = self.sizes[::-1]
        self.dec1 = nn.Sequential(nn.ConvTranspose3d(in_channels = dims[3], out_channels = dims[2], kernel_size = kernel_size, padding = tconv_padding, stride = maxpool_stride, output_padding = output_padding),
                                    nn.BatchNorm3d(dims[2]),nn.ReLU(),
                                    nn.Conv3d(in_channels = dims[2], out_channels = dims[2], kernel_size = kernel_size, stride = conv_stride, padding = kernel_size-2),
                                    nn.BatchNorm3d(dims[2]),nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose3d(in_channels = 2*dims[2], out_channels = dims[1], kernel_size = kernel_size, padding = tconv_padding, stride = maxpool_stride, output_padding = output_padding),
                                    nn.BatchNorm3d(dims[1]),nn.ReLU(),
                                    nn.Conv3d(in_channels = dims[1], out_channels = dims[1], kernel_size = kernel_size, stride = conv_stride, padding = kernel_size-2),
                                    nn.BatchNorm3d(dims[1]),nn.ReLU())
        # self.dec3 = nn.Sequential(nn.ConvTranspose3d(in_channels = 2*dims[1], out_channels = dims[0], kernel_size = kernel_size, padding = tconv_padding, stride = maxpool_stride, output_padding = output_padding),
        #                             nn.ReLU())
        self.final_layer = nn.Sequential(nn.Conv3d(in_channels = 2*dims[1], out_channels = 1, kernel_size = kernel_size, padding = kernel_size-2, stride = conv_stride))
        self.dec_linear_1 = nn.Linear(latent_size, self.sizes[0])
        self.dec_linear_2 = nn.Linear(latent_size, self.sizes[1])
        self.dec_linear_3 = nn.Linear(latent_size, self.sizes[2])
        #self.dec_linear_4 = nn.Linear(latent_size, self.sizes[3])

 
       

        self.latent = dist.Normal(0, 1) # standard normal prior
        self.latent.loc = self.latent.loc.to(device) # hack to get sampling on the GPU
        self.latent.scale = self.latent.scale.to(device)
        #self.kl = 0
        self.mus = []
        self.sigmas = []
        self.to(device)
        self.mu_1 = None
        self.sigma_1 = None
        self.mu_2 = None
        self.sigma_2 = None
        self.mu_3 = None
        self.sigma_3 = None
        

    def forward(self, x):
        c_0 = x.shape[1]
 

        x = self.enc1(x)
        x1_mu, x1_sigma = self.enc_mu_1(x), torch.exp(0.5*self.enc_sigma_1(x))
        self.mu_1 = x1_mu
        self.sigma_1 = x1_sigma
        x1 = x1_mu + x1_sigma*self.latent.sample(x1_mu.shape)
        x = self.maxpool(x)
        x = self.enc2(x)
        x2_mu, x2_sigma = self.enc_mu_2(x), torch.exp(0.5*self.enc_sigma_2(x))
        x2 = x2_mu + x2_sigma*self.latent.sample(x2_mu.shape)
        self.mu_2 = x2_mu
        self.sigma_2 = x2_sigma
        x = self.maxpool(x)
        x = self.enc3(x)
        x3_mu, x3_sigma = self.enc_mu_3(x), torch.exp(0.5*self.enc_sigma_3(x))
        x3 = x3_mu + x3_sigma*self.latent.sample(x3_mu.shape)
        self.mu_3 = x3_mu
        self.sigma_3 = x3_sigma

        x = self.dec_linear_1(x3)
        x = self.dec1(x)
        x = x[:,:,:self.sizes[1],:self.sizes[1],:self.sizes[1]]
        x2 = self.dec_linear_2(x2)
        x = torch.cat((x2, x), 1)
        x = self.dec2(x)
        x = x[:,:,:self.sizes[2],:self.sizes[2],:self.sizes[2]]

        x1 = self.dec_linear_3(x1)
        x = torch.cat((x1, x), 1)

        x = self.final_layer(x)
        #x = x[:,:,:self.sizes[2],:self.sizes[2],:self.sizes[2]]
        return x


    def sample(self,x, n):
        samples = None
        with torch.no_grad():
            c_0 = x.shape[1]
    

            x = self.enc1(x)
            x1_mu, x1_sigma = self.enc_mu_1(x), torch.exp(0.5*self.enc_sigma_1(x))
            self.mu_1 = x1_mu
            self.sigma_1 = x1_sigma
            
            x = self.maxpool(x)
            x = self.enc2(x)
            x2_mu, x2_sigma = self.enc_mu_2(x), torch.exp(0.5*self.enc_sigma_2(x))
            self.mu_2 = x2_mu
            self.sigma_2 = x2_sigma
            x = self.maxpool(x)
            x = self.enc3(x)
            x3_mu, x3_sigma = self.enc_mu_3(x), torch.exp(0.5*self.enc_sigma_3(x))
            self.mu_3 = x3_mu
            self.sigma_3 = x3_sigma


            print("Sampling")
            for _ in range(n):
                x1 = x1_mu + x1_sigma*self.latent.sample(x1_mu.shape)
                x2 = x2_mu + x2_sigma*self.latent.sample(x2_mu.shape)
                x3 = x3_mu + x3_sigma*self.latent.sample(x3_mu.shape)
                
                print(_)

                x = self.dec_linear_1(x3)
                x = self.dec1(x)
                x = x[:,:,:self.sizes[1],:self.sizes[1],:self.sizes[1]]
                x2 = self.dec_linear_2(x2)
                x = torch.cat((x2, x), 1)
                x = self.dec2(x)
                x = x[:,:,:self.sizes[2],:self.sizes[2],:self.sizes[2]]

                x1 = self.dec_linear_3(x1)
                x = torch.cat((x1, x), 1)

                x = self.final_layer(x)

                if samples   != None:
                    samples = torch.cat((samples, x.unsqueeze(0)), 0)
                else:
                    samples = x.unsqueeze(0)
            return samples


def create_datset(path, debug = False):
    n_files = len([img for img in os.listdir(path) if img.startswith('clean_image')])
    clean = None
    noisy = None
    labels = None
    for i in range(n_files):
        clean_img = torch.tensor(nib.load(os.path.join(path, f'clean_image_{i}.nii.gz')).get_fdata()).unsqueeze(0)
        noisy_img = torch.tensor(nib.load(os.path.join(path, f'image_{i}.nii.gz')).get_fdata()).unsqueeze(0)
        label_img = torch.tensor(nib.load(os.path.join(path, f'labels_{i}.nii.gz')).get_fdata()).unsqueeze(0)
        if clean is not None:
            clean = torch.cat((clean, clean_img), dim=0)
            noisy = torch.cat((noisy, noisy_img), dim=0)
            labels = torch.cat((labels, label_img), dim=0)
        else:
            clean = clean_img
            noisy = noisy_img
            labels = label_img
            if debug:
                break

    return clean.unsqueeze(1), noisy.unsqueeze(1), labels.unsqueeze(1)


def train_vae(vae, data, epochs=20, metric = 'bce', regularization = 1e-4, lr = 1e-4, batch_size = 20):
    opt = torch.optim.Adam(vae.parameters(), lr = lr)
    training_loader = DataLoader(list(zip(data[0],data[1])), batch_size=batch_size, shuffle=True)
    #print("{:.3f}MB allocated1".format(torch.cuda.memory_allocated()/1024**2))

    
    if metric == 'mse':
        loss_fn = nn.MSELoss()
    elif metric == 'bce':
        loss_fn = nn.BCELoss()
    else:
        raise ValueError('metric must be "mse" or "bce"')
    std_normal = None
    
    losses = []
    kl_losses = []
    metric_losses = []
    best_loss = np.inf
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        batch_losses = []
        kl_batch_losses = []
        metric_batch_losses = []
        for i, (X_batch, y_batch) in enumerate(training_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            #print("{:.3f}MB allocated2".format(torch.cuda.memory_allocated(device)/1024**2))

            
            x_hat = vae(X_batch)
            
            #computing latent_loss
            kl_1 = kl_div(dist.Normal(vae.mu_1, vae.sigma_1), dist.Normal(torch.zeros_like(vae.mu_1),torch.ones_like(vae.sigma_1))).sum()
            kl_2 = kl_div(dist.Normal(vae.mu_2, vae.sigma_2), dist.Normal(torch.zeros_like(vae.mu_2),torch.ones_like(vae.sigma_2))).sum()
            kl_3 = kl_div(dist.Normal(vae.mu_3, vae.sigma_3), dist.Normal(torch.zeros_like(vae.mu_3),torch.ones_like(vae.sigma_3))).sum()
            #print("{:.3f}MB allocated2.2".format(torch.cuda.memory_allocated(device)/1024**2))


         
            #kl_loss = torch.tensor(0)
            # print(kl_loss)
            #Potential bug - does this account for all mu and sigma, or just the last one?
            #kl_loss = kl_div(dist.Normal(vae.mu, vae.sigma), std_normal).sum()
            #print(f'kl_loss: {kl_loss}, loss: {loss_fn(x_hat, x)}')
            loss = loss_fn(x_hat, X_batch) + regularization*(kl_1 + kl_2 + kl_3)
            #print("{:.3f}MB allocated3".format(torch.cuda.memory_allocated(device)/1024**2))

            loss.backward()
            #print("{:.3f}MB allocated4".format(torch.cuda.memory_allocated(device)/1024**2))

            opt.step()
            #print("{:.3f}MB allocated5".format(torch.cuda.memory_allocated(device)/1024**2))

            opt.zero_grad(set_to_none=True)
            #print("{:.3f}MB allocated6".format(torch.cuda.memory_allocated(device)/1024**2))


            if i % 10 == 0:

                print(i, epoch, loss.detach().item())
                print(f'{i}, {epoch}, {loss.detach().item()}kl_1: {kl_1.detach().item()}, kl_2: {kl_2.detach().item()}, kl_3: {kl_3.detach().item()}')
            #print("{:.3f}MB allocated7".format(torch.cuda.memory_allocated(device)/1024**2))
            batch_losses.append(loss.detach().item())
            kl_batch_losses.append((kl_1.detach().item()+kl_2.detach().item()+kl_3.detach().item())*regularization)
            metric_batch_losses.append(loss_fn(x_hat, X_batch).detach().item())
            torch.cuda.empty_cache()
            gc.collect()
            #print("{:.3f}MB allocated8".format(torch.cuda.memory_allocated(device)/1024**2))
        print(f'Epoch {epoch} loss: {np.mean(batch_losses)}')
        print(f'Epoch {epoch} kl loss: {np.mean(kl_batch_losses)}')
        losses.append(np.mean(batch_losses))
        kl_losses.append(np.mean(kl_batch_losses))
        metric_losses.append(np.mean(metric_batch_losses))
    plt.plot(losses, label='total loss')
    plt.plot(kl_losses, label='kl loss')
    plt.plot(metric_losses, label='metric loss')
    plt.legend()
    return losses, kl_losses, metric_losses, vae


def test_output(vae, noisy,clean, output_folder):

    for i in [6,30,36,50]:
            
        test = vae(noisy[i:i+1,:,:,:].to(device))
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(noisy[i, 0, 80,:, :].detach().cpu().numpy(), cmap='gray')
        axs[0].set_title('Noisy Input')
        axs[1].imshow(test[0, 0, 80,:, :].detach().cpu().numpy(), cmap='gray')
        axs[1].set_title('Test Output')

        axs[2].imshow(clean[i, 0, 80,:, :].detach().cpu().numpy(), cmap='gray')
        axs[2].set_title('Clean Input')
        plt.savefig(os.path.join(output_folder, f'test_output_{i}.png'))


def run(param_file, data_folder, output_folder, debug = False):

    clean, noisy, labels = create_datset(data_folder, debug)

    noisy = noisy.type(torch.float32)
    clean = clean.type(torch.float32)

    noisy = (noisy - noisy.mean())/noisy.std()
    clean = (clean - clean.mean())/clean.std()


    with open(param_file, "r") as f:
        params = json.load(f)


    vae = simple_VAE(image_size=noisy.shape[1:], device = device, dims = params['dims'], latent_size= params['latent_size'],  kernel_size = params['kernel_size'], maxpool_stride = params['maxpool_stride'])
    losses, kl_losses, metric_losses, vae = train_vae(vae, [noisy, clean], epochs = params["epochs"], metric='mse', regularization=params["kl_beta"], lr = params["lr"], batch_size = params["batch_size"])
    plt.plot(losses, label='total loss')
    plt.plot(kl_losses, label='kl loss')
    plt.plot(metric_losses, label='metric loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'losses.png'))

    torch.save(vae.state_dict(), f"{output_folder}/{params["model_name"]}.pth")
    test_output(vae, noisy, clean, output_folder)
    print("Remember beta is not implemented for hierarchical VAEs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a VAE model.')
    parser.add_argument('--param_file', type=str, required=True, help='Path to the parameter file.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
    parser.add_argument('--debug', type=int,default = 0, help='1 for Debug mode.')
    args = parser.parse_args()

    param_file = args.param_file
    data_folder = args.data_folder
    output_folder = args.output_folder
    debug = args.debug
    
    if debug == 1:
        debug = True
    else:
        debug = False

    device = torch.device(f"cuda:{str(args.gpu)}") if torch.cuda.is_available() else 'cpu'
    print(device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"


    run(param_file, data_folder, output_folder, debug)
