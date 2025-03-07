# U-Net with injection of VAE in second final layer
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
import pandas as pd

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

# device = torch.device("cuda:4")
# print(device)
# # os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def ECE(y_hat, y_true, n_bins, plot = False):
    
    # compute bins
    bins = torch.linspace(0,1,n_bins+1).to(y_hat.device)
    ece = 0
    shape = y_hat.shape
    y_hat = torch.flatten(y_hat)
    y_true = torch.flatten(y_true)
    positives = []
    for i in range(n_bins):
        # Divide into bins
        bin_prob = y_hat[(y_hat >= bins[i]) & (y_hat < bins[i+1])]
        bin_true = y_true[(y_hat >= bins[i]) & (y_hat < bins[i+1])]
        positives.append(torch.sum(bin_true)/bin_true.shape[0])
        #3. compute bin accuracy
        bin_acc = torch.sum((bin_true-bin_prob)>=0.5)/bin_prob.shape[0]
        
        #4. compute bin confidence
        bin_conf = torch.mean(bin_prob)
        
        #5. compute bin ECE
        ece += (bin_prob.shape[0]*torch.abs(bin_acc - bin_conf))/shape[0]
    # plot histogram
    if plot:
        plt.plot(bins[1:], positives)
        plt.show()
    return ece


def ACE(y_hat, y_true, n_bins, plot = False):
    
    # compute bins
    bins = torch.linspace(0,1,n_bins+1).to(y_hat.device)
    ece = 0
    shape = y_hat.shape
    y_hat = torch.flatten(y_hat)
    y_true = torch.flatten(y_true)
    positives = []
    non_empty_bins = 0
    for i in range(n_bins):
        # Divide into bins
        bin_prob = y_hat[(y_hat >= bins[i]) & (y_hat < bins[i+1])]
        if bin_prob.shape[0] != 0:
            
            non_empty_bins += 1
            bin_true = y_true[(y_hat >= bins[i]) & (y_hat < bins[i+1])]
            positives.append(torch.sum(bin_true)/bin_true.shape[0])
            #3. compute bin accuracy
            bin_acc = torch.sum((bin_true-bin_prob)>=0.5)/bin_prob.shape[0]
            
            #4. compute bin confidence
            bin_conf = torch.mean(bin_prob)
            
            #5. compute bin ECE
            ece += torch.abs(bin_acc - bin_conf)
    # plot histogram
    if plot:
        plt.plot(bins[1:], positives)
        plt.show()
    return ece

def reliability_diagram(y_hat, y_true, n_bins, plot = False):
    
    # compute bins
    bins = torch.linspace(0,1,n_bins+1).to(y_hat.device)
    ece = 0
    shape = y_hat.shape
    y_hat = torch.flatten(y_hat)
    y_true = torch.flatten(y_true)

    positives = []
    #numbers = []
    for i in range(n_bins):
        # Divide into bins
        #bin_prob = y_hat[(y_hat >= bins[i]) & (y_hat < bins[i+1])]
        bin_true = y_true[(y_hat >= bins[i]) & (y_hat < bins[i+1])]
        positives.append(torch.sum(bin_true).item())
        #3. compute bin accuracy
    if plot == True:
        plt.bar(np.arange(n_bins), np.array(positives))
        plt.xticks(np.arange(n_bins), [str(i) for i in bins[1:]])
        
        plt.show()
    return bins, positives


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        upper = 2*torch.sum(y_pred*y_true) + 1
        lower = torch.sum(y_pred) + torch.sum(y_true) + 1
        return 1 - upper/lower



class doubleconv(nn.Module):
    # Module for the double convolution block
    def __init__(self, in_channels,middle_channels, out_channels):
        super(doubleconv, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv3d(in_channels = in_channels, out_channels = middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels = middle_channels, out_channels = out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU())

    def forward(self, x):
        return self.sequence(x)



class simple_VAE(nn.Module):
    def __init__(self, image_size,  device, dims, latent_size, kernel_size = 5, maxpool_stride = 1, conv_stride = 1, predictive = False):
        super(simple_VAE, self).__init__()
        self.image_size = torch.prod(torch.tensor(image_size)).item()
        self.predictive = predictive
        self.out_channels = 2*dims[1]
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

        #x = self.final_layer(x)
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

                #x = self.final_layer(x)

                if samples   != None:
                    samples = torch.cat((samples, x.unsqueeze(0)), 0)
                else:
                    samples = x.unsqueeze(0)
            return samples


class probnet(nn.Module):
    def __init__(self, image_size = 60, pretrained_vae = None, device = "cpu"):
        super(probnet, self).__init__()
        channels = [1, 16, 32, 64, 128, 256, 512]
        self.image_size = image_size
        self.vae = pretrained_vae

        #Encoder
        self.double_1 = doubleconv(channels[0], channels[1], channels[2])
        self.double_2 = doubleconv(channels[2], channels[2], channels[3])
        self.double_3 = doubleconv(channels[3], channels[3], channels[4])
        self.double_4 = doubleconv(channels[4], channels[4], channels[5])
        self.double_5 = doubleconv(channels[5], channels[5], channels[6])
        self.double_6 = doubleconv(channels[6] + channels[5], channels[5], channels[5])
        self.double_7 = doubleconv(channels[5]+channels[4], channels[4], channels[4])

        self.double_8 = doubleconv(channels[4]+channels[3], channels[3], channels[3])
        # self.double_9 = doubleconv(channels[3]+channels[2]+self.vae.out_channels, channels[2], channels[2])
        self.double_9 = doubleconv(channels[3]+self.vae.out_channels, channels[2], channels[2])

        self.maxpool = nn.MaxPool3d(kernel_size=2)

        self.conv8 = nn.Conv3d(in_channels=channels[2], out_channels=channels[0], kernel_size=1, padding=0)   #32->1
        self.tconv1 = nn.ConvTranspose3d(in_channels=channels[-1], out_channels=channels[-1], kernel_size=3, stride=2, padding=0)
        self.tconv2 = nn.ConvTranspose3d(in_channels=channels[-2], out_channels=channels[-2], kernel_size=3, stride=2, padding=0)
        self.tconv3 = nn.ConvTranspose3d(in_channels=channels[-3], out_channels=channels[-3], kernel_size=3, stride=2, padding=0)
        self.tconv4 = nn.ConvTranspose3d(in_channels=channels[-4], out_channels=channels[-4], kernel_size=3, stride=2, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.to(device)
    
    def forward(self, x, x_vae):
        #encoder
        c_0 = self.image_size
        c_1  = c_0//2
        c_2 = c_1//2
        c_3 = c_2//2
        
        #Down
        x_vae = self.vae(x)
        x1 = self.double_1(x)
        x = self.maxpool(x1)
        x2 = self.double_2.forward(x)
        x = self.maxpool(x2)
        x3 = self.double_3.forward(x)
        x = self.maxpool(x3)
        x4 = self.double_4.forward(x)

        x = self.maxpool(x4)
        x = self.double_5.forward(x)


        #Up
        x = self.tconv1(x)
        x = x[:, :,  :c_3, :c_3, :c_3]
        x = torch.cat([x4, x], dim=1)
        x = self.double_6.forward(x)


        x = self.tconv2(x)
        x = x[:, :,  :c_2, :c_2, :c_2]
        x = torch.cat([x3, x], dim=1)
        x = self.double_7.forward(x)
        x = self.tconv3(x)
        x = x[:, :,  :c_1, :c_1, :c_1]
        x = torch.cat([x2, x], dim=1)
        x = self.double_8.forward(x)
        x = self.tconv4(x)
        x = x[:, :,  :c_0, :c_0, :c_0]
        x = torch.cat([ x, x_vae], dim=1)

        #x = torch.cat([x1, x, x_vae], dim=1)
        x = self.double_9.forward(x)
        x = self.conv8(x)
        x = self.sigmoid(x)
        

        return x
    
    def sample(self, x, n_samples):
        #encoder
        c_0 = self.image_size
        c_1  = c_0//2
        c_2 = c_1//2
        c_3 = c_2//2
        samples = None
        #Down
        vae_samples = self.vae.sample(x, n_samples)
        x1 = self.double_1(x)
        x = self.maxpool(x1)
        x2 = self.double_2.forward(x)
        x = self.maxpool(x2)
        x3 = self.double_3.forward(x)
        x = self.maxpool(x3)
        x4 = self.double_4.forward(x)

        x = self.maxpool(x4)
        x = self.double_5.forward(x)


        #Up
        x = self.tconv1(x)
        x = x[:, :, :c_3, :c_3, :c_3]
        x = torch.cat([x4, x], dim=1)
        x = self.double_6.forward(x)


        x = self.tconv2(x)
        x = x[:, :, :c_2, :c_2, :c_2]
        x = torch.cat([x3, x], dim=1)
        x = self.double_7.forward(x)
        x = self.tconv3(x)
        x = x[:, :,  :c_1, :c_1, :c_1]
        x = torch.cat([x2, x], dim=1)
        x = self.double_8.forward(x)
        x = self.tconv4(x)
        x_i = x[:, :,  :c_0, :c_0, :c_0]
        for i in range(n_samples):
            # x = torch.cat([x1, x_i, vae_samples[i]], dim=1)
            x = torch.cat([ x_i, vae_samples[i]], dim=1)
            x = self.double_9.forward(x)
            x = self.conv8(x)
            x = self.sigmoid(x)
            if samples != None:
                samples = torch.cat([samples, x],dim= 0)
            else:
                samples = x
            

        return samples
    


def create_datset(path, debug = False, prob = True):
    n_files = len([img for img in os.listdir(path) if img.startswith('clean_image')])
    clean = None
    noisy = None
    labels = None
    for i in range(n_files):
        clean_img = torch.tensor(nib.load(os.path.join(path, f'clean_image_{i}.nii.gz')).get_fdata()).unsqueeze(0)
        noisy_img = torch.tensor(nib.load(os.path.join(path, f'image_{i}.nii.gz')).get_fdata()).unsqueeze(0)
        label_img = torch.tensor(nib.load(os.path.join(path, f'labels_{i}.nii.gz')).get_fdata()).unsqueeze(0)
        if prob:
            label_img = label_img == 47
            label_img = label_img.int().type(torch.float32)
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


def train_probnet(net, pretrained_vae, data, epochs=20, metric = 'bce', regularization = 1e-4, lr = 1e-4, batch_size = 20):
    opt = torch.optim.Adam(net.parameters(), lr = lr)
    training_loader = DataLoader(list(zip(data[0],data[1])), batch_size=batch_size, shuffle=True)
    #print("{:.3f}MB allocated1".format(torch.cuda.memory_allocated()/1024**2))

    
    if metric == 'bce':
        loss_fn = nn.BCELoss()
    elif metric == 'Dice':
        loss_fn = DiceLoss()
        #loss_fn = nn.BCELoss()
    else:
        raise ValueError('metric must be "mse" or "bce"')
    std_normal = None
    
    losses = []
    #kl_losses = []
    #metric_losses = []
    best_loss = np.inf
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        batch_losses = []
        #kl_batch_losses = []
        #metric_batch_losses = []
        for i, (X_batch, y_batch) in enumerate(training_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            #print("{:.3f}MB allocated2".format(torch.cuda.memory_allocated(device)/1024**2))

            # Run the VAE
            with torch.no_grad():
                x_hat = pretrained_vae(X_batch)

            #predict
            y_hat = net(X_batch, x_hat)

          
            loss = loss_fn(y_hat, y_batch) 
            #print("{:.3f}MB allocated3".format(torch.cuda.memory_allocated(device)/1024**2))

            loss.backward()
            #print("{:.3f}MB allocated4".format(torch.cuda.memory_allocated(device)/1024**2))

            opt.step()
            #print("{:.3f}MB allocated5".format(torch.cuda.memory_allocated(device)/1024**2))

            opt.zero_grad(set_to_none=True)
            #print("{:.3f}MB allocated6".format(torch.cuda.memory_allocated(device)/1024**2))


            if i % 10 == 0:

                #print(i, epoch, loss.detach().item())
                print(f'{i}, {epoch+1}, {loss.detach().item()}')
            #print("{:.3f}MB allocated7".format(torch.cuda.memory_allocated(device)/1024**2))
            batch_losses.append(loss.detach().item())
            torch.cuda.empty_cache()
            gc.collect()
            #print("{:.3f}MB allocated8".format(torch.cuda.memory_allocated(device)/1024**2))
        print(f'Epoch {epoch} loss: {np.mean(batch_losses)}')
        losses.append(np.mean(batch_losses))

    plt.plot(losses, label='total loss')

    plt.legend()
    return losses, net


def test_output(net, noisy,labels, output_folder):

    for i in [6,30,36,50]:
            
        test = net.sample((noisy[i:i+1,:,:,:].to(device)),1)
        test = test[0,0,:,:,:].detach().cpu().numpy()
        nib.save(nib.Nifti1Image(test, np.eye(4)), os.path.join(output_folder, f'test_output_{i}.nii.gz'))
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(noisy[i, 0, 80,:, :].detach().cpu().numpy(), cmap='gray')
        axs[0].set_title('Input')
        axs[1].imshow(test[80,:,:], cmap='gray')
        axs[1].set_title('Test Output')

        axs[2].imshow(labels[i, 0, 80,:, :].detach().cpu().numpy(), cmap='gray')
        axs[2].set_title('Label')
        plt.savefig(os.path.join(output_folder, f'test_output_{i}.png'))


def run_prediction(net, noisy,labels, output_folder, n_samples = 5):
    df = pd.DataFrame(columns = ['Dice','ECE', 'ACE', 'Reliability Diagram'])
    dl = DiceLoss()
    for i in range(noisy.shape[0]):
        with torch.no_grad():
            test = net.sample((noisy[i:i+1,:,:,:].to(device)),n_samples)
        print(test.shape)
        test = torch.mean(test,dim=0).unsqueeze(0).cpu()
        label = labels[i:i+1,:,:,:].long().cpu()
        dice = 1 - dl(test, label).item()
        ace = ACE(test, label, 10).item()
        ece = ECE(test, label, 10).item()
        bins, positives = reliability_diagram(test, label, 10)
        df.loc[i] = [dice,ece, ace, positives]
        


        test = test[0,0,:,:,:].detach().cpu().numpy()
        label = labels[i,0,:,:,:].detach().cpu().numpy()
        nib.save(nib.Nifti1Image(test, np.eye(4)), os.path.join(output_folder, f'test_output_{i}.nii.gz'))
        nib.save(nib.Nifti1Image(label, np.eye(4)), os.path.join(output_folder, f'label_{i}.nii.gz'))

    df.to_csv(os.path.join(output_folder, 'results.csv'))


def run(param_file, data_folder, output_folder,predict = False, debug = False):

    clean, noisy, labels = create_datset(data_folder, debug)

    noisy = noisy.type(torch.float32)
    clean = clean.type(torch.float32)

    noisy = (noisy - noisy.mean())/noisy.std()
    clean = (clean - clean.mean())/clean.std()



    with open(param_file, "r") as f:
        params = json.load(f)

    

    vae = simple_VAE(image_size=noisy.shape[1:], device = device, dims = params['dims'], latent_size= params['latent_size'],  kernel_size = params['kernel_size'], maxpool_stride = params['maxpool_stride'])
    vae.load_state_dict(torch.load(params['pretrained_vae'], map_location = device))
    for vae_param in vae.parameters():
        vae_param.requires_grad = False
    net = probnet(image_size = int(noisy.shape[2]), pretrained_vae = vae, device = device)
    if predict:
        net.load_state_dict(torch.load(params['pretrained_net'], map_location = device))
        for net_param in net.parameters():
            net_param.requires_grad = False
        run_prediction(net, noisy, labels, output_folder, params['n_samples'])
        return
    losses, net = train_probnet(net, vae, [noisy, labels], epochs = params["epochs"], metric='Dice', lr = params["lr"], batch_size = params["batch_size"])
    plt.plot(losses, label='total loss')
    plt.legend()
    plt.savefig(os.path.join(output_folder, 'losses.png'))

    torch.save(net.state_dict(), f"{output_folder}/{params["model_name"]}.pth")
    #test_output(net, noisy, labels, output_folder)
    print("Remember to test with multichannel output from VAE")
    #print("Remember beta is not implemented for hierarchical VAEs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a probnet.')
    parser.add_argument('--param_file', type=str, required=True, help='Path to the parameter file.')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the data folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the output folder.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU number to use.')
    parser.add_argument('--predict', type=int,default = 0, help='1 for prediciton mode.')
    parser.add_argument('--debug', type=int,default = 0, help='1 for Debug mode.')
    args = parser.parse_args()

    param_file = args.param_file
    data_folder = args.data_folder
    output_folder = args.output_folder
    debug = args.debug
    predict = args.predict
    
    if debug == 1:
        debug = True
    else:
        debug = False
    if predict == 1:
        predict = True
    else:
        predict = False

    device = torch.device(f"cuda:{str(args.gpu)}") if torch.cuda.is_available() else 'cpu'
    print(device)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"


    run(param_file, data_folder, output_folder, predict, debug)


