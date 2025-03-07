import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib


def ECE(y_hat, y_true, n_bins, plot = False):
    #Compute the Expected Calibration Error
    # compute bins
    bins = np.linspace(0,1,n_bins+1)
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
    #Compute the Average Calibration Error
    # compute bins
    bins = np.linspace(0,1,n_bins+1)
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
    #Compute the reliability diagram
    # compute bins
    bins = np.linspace(0,1,n_bins+1)
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
        positives.append(torch.sum(bin_true).item()/bin_true.shape[0])
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


def run_analysis(folder):
    #run analysis on the results of the experiments
    df = pd.DataFrame(columns = ['Dice','ECE', 'ACE', 'Reliability Diagram'])
    dl = DiceLoss()

    for i in range(0,40):
        print(i)
        test = nib.load(f"{folder}/test_output_{i}.nii.gz").get_fdata()
        labels = nib.load(f"{folder}/label_{i}.nii.gz").get_fdata()
        test = torch.tensor(test)
        labels = torch.tensor(labels).long()
        dice = 1 - dl(test, labels).item()
        ace = ACE(test, labels, 10).item()
        ece = ECE(test, labels, 10).item()
        bins, positives = reliability_diagram(test, labels, 10)
        df.loc[i] = [dice,ece, ace, positives]
    df.to_csv(f"{folder}/results_new.csv")



def resampling_test(folder, n_samples, bins, seed = 42):
    #Resampling test for q0.05, q0.95 by Bröcker and Smith
    #sample test data
    sample_idxs = np.random.uniform(0,40,n_samples).astype(int)

    eces = None
    aces = None
    for i in range(n_samples):
        print(i)

        # bootstrap sample of probabilities

        sample = torch.tensor(nib.load(f"{folder}/test_output_{sample_idxs[i]}.nii.gz").get_fdata())


        # true labels
        #labels_t = y_true[sample_idxs[i]]

        #1 sample labels
        dist = torch.distributions.Bernoulli(sample)
        labels_h = dist.sample()

        #2 compute ECE
        ece = ECE(sample, labels_h, bins).unsqueeze(0)
        ace = ACE(sample, labels_h, bins).unsqueeze(0)
        b, positive = reliability_diagram(sample, labels_h, bins)
        positive = torch.tensor(positive).unsqueeze(0)

        if eces != None:
            eces = torch.cat((eces,ece),0)
            aces = torch.cat((aces,ace),0)
            positives = torch.cat((positives, positive),0)
        else:
            eces = ece
            aces = ace
            positives = positive



    return aces,eces, positives




resampling = {}

for ex in ['ex1_1', 'ex1_2', 'ex2_1', 'ex2_2', 'ex3_1', 'ex3_2']:
    print(ex)
    #run_analysis(f"/Users/aksel/Code/probnet/probnet/experiments/{ex}")
    aces, eces,positives = resampling_test(f"{ex}", 100, 10)
    resampling[f'{ex}_aces'] = aces
    resampling[f'{ex}_eces'] = eces
    resampling[f'{ex}_positives'] = positives

fig, axes = plt.subplots(3, 2, figsize=(15, 15))



#PLOTTING HISTOGRAMS


for i in ['ex1_1', 'ex1_2', 'ex2_1', 'ex2_2', 'ex3_1', 'ex3_2']:
    aces = resampling[f'{i}_eces'].numpy()
    f,s = int(i[2])-1,int(i[-1])-1
    print(f,s)
    ax = axes[f,s]
    ax.hist(aces, bins=20, edgecolor='black')
    ax.hist(aces, bins=20, edgecolor='black')
    if s == 1 and f == 0:
        ax.set_title(f'Straight U-Net\n$\\beta = 1e-7$')
    elif s == 0 and f == 0:
        ax.set_title(f'Multichannel U-Net\n$\\beta = 1e-7$')

    else:
        if f == 0:
            ax.set_title(r'$\beta = 1e-7$')
        elif f == 1:
            ax.set_title(r'$\beta = 1e-6$')
        else:
            ax.set_title(r'$\beta = 1e-8$')
    ax.set_ylabel('Frequency')

fig.title(f'Histogram of ECEs')


plt.tight_layout()
plt.show()

res_1_1 = pd.read_csv("experiments/ex1_1/results_new.csv")
res_1_1["experiment"] = "ex1_1"
res_1_2 = pd.read_csv("experiments/ex1_2/results_new.csv")
res_1_2["experiment"] = "ex1_2"
res_3_1 = pd.read_csv("experiments/ex3_1/results_new.csv")
res_3_1["experiment"] = "ex3_1"
res_2_1 = pd.read_csv("experiments/ex2_1/results_new.csv")
res_2_1["experiment"] = "ex2_1"
res_2_2 = pd.read_csv("experiments/ex2_2/results_new.csv")
res_2_2["experiment"] = "ex2_2"
res_3_2 = pd.read_csv("experiments/ex3_2/results_new.csv")
res_3_2["experiment"] = "ex3_2"




combined = pd.concat([res_1_1, res_1_2, res_2_1, res_2_2,res_3_1, res_3_2], axis = 0)

combined['Reliability Diagram'] = combined['Reliability Diagram'].apply(lambda x: np.fromstring(x.strip('[]'), sep=', '))

reliability_diagram_cols = pd.DataFrame(combined['Reliability Diagram'].tolist(), index=combined.index)
reliability_diagram_cols.columns = [f'Reliability Diagram {i+1}' for i in range(reliability_diagram_cols.shape[1])]
combined = pd.concat([combined, reliability_diagram_cols], axis=1)

combined = combined.rename(columns=lambda x: x.replace('Reliability Diagram', 'RD'))

combined = combined.drop(columns=['RD'])

avgs = combined.groupby("experiment").mean()


n_bins = 10
bins = np.linspace(0,1,n_bins+1).round(2)


fig, axes = plt.subplots(3, 1, figsize=(10, 15))

experiments = ['ex1_1', 'ex2_1', 'ex3_1']

for ax, experiment in zip(axes, experiments):
    rd_columns = [col for col in combined.columns if col.startswith('RD')]
    rd_array = avgs[avgs.index == experiment][rd_columns].to_numpy()
    q_5 = resampling[f'{experiment}_positives'].quantile(0.05, axis = 0)
    q_95 = resampling[f'{experiment}_positives'].quantile(0.95, axis = 0)
    ax.fill_between(np.arange(n_bins), q_5, q_95, alpha = 0.5)
    ax.bar(np.arange(n_bins) - 0.5, rd_array[0], width=0.8)
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels(bins[1:], rotation=45, ha='right')
    ax.set_title(f'Reliability Diagram for {experiment}')
    ax.set_xlabel('Bins')
    ax.set_ylabel('Frequency')
    ax.set_ylim(0, 1)  # Set y-axis limits between 0 and 1
fig.suptitle('Reliability Diagrams for straight U-Net', fontsize=16)
axes[0].set_title(r'$\beta = 1e-7$')
axes[1].set_title(r'$\beta = 1e-6$')
axes[2].set_title(r'$\beta = 1e-8$')

plt.tight_layout()
plt.show()




fig, axes = plt.subplots(3, 1, figsize=(10, 15))

experiments = ['ex1_2', 'ex2_2', 'ex3_2']

for ax, experiment in zip(axes, experiments):
    rd_columns = [col for col in combined.columns if col.startswith('RD')]
    rd_array = avgs[avgs.index == experiment][rd_columns].to_numpy()
    q_5 = resampling[f'{experiment}_positives'].quantile(0.05, axis = 0)
    q_95 = resampling[f'{experiment}_positives'].quantile(0.95, axis = 0)
    ax.fill_between(np.arange(n_bins), q_5, q_95, alpha = 0.5)
    ax.bar(np.arange(n_bins) - 0.5, rd_array[0], width=0.8)
    ax.set_xticks(np.arange(n_bins))
    ax.set_xticklabels(bins[1:], rotation=45, ha='right')
    ax.set_title(f'Reliability Diagram for {experiment}')
    ax.set_xlabel('Bins')
    ax.set_ylabel('Frequency')
    ax.set_ylim(0, 1)  # Set y-axis limits between 0 and 1
fig.suptitle('Reliability Diagrams for multichannel U-Net', fontsize=16)
axes[0].set_title(r'$\beta = 1e-7$')
axes[1].set_title(r'$\beta = 1e-6$')
axes[2].set_title(r'$\beta = 1e-8$')

plt.tight_layout()
plt.show()


# Create a table with all the averages except the ones with RD
avg_table = avgs.drop(columns=[col for col in avgs.columns if col.startswith('RD')])
print(avg_table)

# Create a table with all the averages except the ones with RD
avg_table = avgs.drop(columns=[col for col in avgs.columns if col.startswith('RD')])
print(avg_table)

# Save the averages to a CSV file
avg_table.to_csv("experiments/averages.csv")

# Convert the averages table to a LaTeX table
latex_table = avg_table.to_latex()
with open("experiments/averages.tex", "w") as f:
    f.write(latex_table)