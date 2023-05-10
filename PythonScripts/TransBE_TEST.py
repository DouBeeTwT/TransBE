import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
torch.set_printoptions(precision=10)
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('Agg')

Dim = 16
PCA_Size = 50
Batch_Size = 256
alpha = 0.8
seed = 42
Max_No_Improve_Steps = 500
torch.cuda.set_device(1)

if os.path.exists("../Figures/Figures_seed"+str(seed)):
    pass
else:
    os.mkdir("../Figures/Figures_seed"+str(seed))

lossfile = open("../Figures/Figures_seed"+str(seed)+"/LossFile.csv",'a')

def Setup_seed(x):
    torch.manual_seed(x)
    torch.cuda.manual_seed(x)
    torch.backends.cudnn.deterministic = True
    np.random.seed(x)

def Classification(filepath):
    adata = sc.read_h5ad(filepath)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_cells(adata, max_genes=30000)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    Data = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata,resolution=0.95)
    
    Leiden = pd.DataFrame(adata.obs["leiden"], index=adata.obs_names)
    Data = pd.concat([Data, Leiden], axis=1, join="inner")
    return Data

def Calculate_PCA(Data, N):
    pca = PCA(n_components=N)
    pca_results = pca.fit(Data)
    D = pca.fit_transform(Data)
    return D

def Calculate_Distance(A, B):
    m = A.shape[0]
    n = B.shape[0]
    D = torch.zeros([m ,n])

    M = torch.matmul(A, B.T)
    H1 = torch.sum(torch.square(A), axis=1).reshape(1,-1)
    H2 = torch.sum(torch.square(B), axis=1).reshape(1,-1)
    D = torch.sqrt(-2*M + H2 + H1.T)
    return D

def Calculate_Normalize(Matrix):
    # Normalize
    for i, index in enumerate(range(Matrix.shape[0])):
        Line = Matrix[index,:]
        Min = torch.min(Line)
        Max = torch.max(Line)
        Matrix[index,:] = (Line - Min) / (Max-Min)
    return Matrix

def Calculate_Loss(D1, D2):
    Distance_Matrix = Calculate_Distance(D1, D2)
    Distance_Matrix = Calculate_Normalize(Distance_Matrix)
    Leiden1 = torch.FloatTensor(list(map(int, Data1["leiden"].to_list()))).cuda().reshape(-1,1)
    #Leiden2 = torch.FloatTensor(list(map(int, Data2["leiden"].to_list())))[0:Batch_Size]
    #Leiden2 = torch.cat((Leiden2, torch.tensor([-1])), 0).cuda().reshape(1,-1)
    Distance_Matrix = torch.cat((Distance_Matrix, Leiden1), 1)
    #Distance_Matrix = torch.cat((Distance_Matrix, Leiden2), 0)

    i_max = Distance_Matrix.shape[0] - 1
    j_max = Distance_Matrix.shape[1] - 1
    Loss_list = torch.zeros([j_max+1]).cuda()
    Distance_Min_Index = torch.argmin(Distance_Matrix[:-1,:],dim=0)

    for i,Index in enumerate(Distance_Min_Index):
        Cluster1 = Distance_Matrix[Index, -1]
        Mask1 = (Distance_Matrix[:,-1] == Cluster1)[:-1]
        Distance_List = Distance_Matrix[:-1,i]
        Distance_In_Group = torch.mean(Distance_List[Mask1])
        Distance_Out_Group = torch.mean(Distance_List[Mask1 == False])
        Loss_list[i] = Distance_In_Group/Distance_Out_Group
    return torch.tensor(torch.sum(Loss_list), requires_grad=False)


class TransBE(nn.Module):
    def __init__(self, Dim=128, header=4) -> None:
        super(TransBE, self).__init__()
        # 先将读入的 1 x 50 的数据经过一个Linear层生成 128 x 50的数据
        self.linear_layer1 = nn.Linear(1,Dim)
        self.relu_layer1 = nn.ReLU()
        # self.drop_layer1 = nn.Dropout(0.1) 不知道要不要加，加载哪里
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=Dim, nhead=header)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=Dim , nhead=header)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
        self.linear_layer2 = nn.Linear(Dim, 1)
        #self.relu_layer2 = nn.ReLU()
    def forward(self, data):
        src = self.relu_layer1(self.linear_layer1(data))
        mid = self.transformer_encoder(src)
        out = self.transformer_decoder(src, mid)
        out = self.linear_layer2(out)
        
        return mid, out

def Plot_Tsne(PCA1, PCA2, file_out_path="./", seed=42):
    tsne1 = TSNE(n_components=2, init='pca', random_state=seed)
    result1 = tsne1.fit_transform(PCA1)
    tsne2 = TSNE(n_components=2, init='pca', random_state=seed)
    result2 = tsne2.fit_transform(PCA2)

    plt.scatter(result1[:,0],result1[:,1], color="hotpink", s=1)
    plt.scatter(result2[:,0],result2[:,1], color="#88c999", s=1)
    plt.savefig(file_out_path)
    plt.close()


time0 = time.time()
Setup_seed(seed)
print("Step 1: ", end="")
Data1, Data2 = Classification("../Data/Sample1.h5ad"), Classification("../Data/Sample2.h5ad")
Keys = list(set(Data1.columns) & set(Data2.columns))
Data1, Data2 = Data1[Keys], Data2[Keys]
Data1, Data2 = Data1.sort_values("leiden"), Data2.sort_values("leiden")
time1 = time.time()
print("%.2f s"%(time1-time0))

print("Step 2: ", end="")
Data_All = pd.concat([Data1.drop(["leiden"], axis=1), Data2.drop(["leiden"], axis=1)])
Data_All = Calculate_PCA(Data_All, N=PCA_Size)
D1 = torch.FloatTensor(Data_All[0:Data1.shape[0], :]).cuda()
D2 = torch.FloatTensor(Data_All[Data1.shape[0]:, :]).cuda()
time2 = time.time()
print("%.2f s"%(time2-time1))

print("Step 3: ", end="")
Gene = []
for i in range(PCA_Size):
    Gene.append(i)
Gene = Variable(torch.LongTensor(Gene))
Gene = torch.tensor(Gene).cuda()

dataloader = DataLoader(TensorDataset(D2), batch_size=Batch_Size, shuffle=True)
model = TransBE(Dim).cuda()
#model = torch.load("../Model/Model_v0.0")
criteria = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters())
time3 = time.time()
print("%.2f s"%(time3-time2))

print("START!")
Loss_Min = 100000
No_Improve_Steps = 0
for epoch in range(10000):
    Total_Loss = 0
    D2_remove_batch = torch.tensor([]).cuda()
    for step, data in enumerate(tqdm(dataloader, leave=False)):
        optimizer.zero_grad()
        ipt = data[0].unsqueeze(2)
        mid, tgt = model(ipt)
        loss1 = criteria(ipt.contiguous().view(-1, ipt.size(-1)), tgt.contiguous().view(-1, tgt.size(-1)))/Batch_Size
        loss2 = Calculate_Loss(D1, torch.squeeze(tgt, 2))/Batch_Size
        loss = alpha*loss1+(1-alpha)*loss2
        loss.backward()
        optimizer.step()
        Total_Loss += loss
        D2_remove_batch = torch.cat((D2_remove_batch, tgt.squeeze(2)), 0)
    
    if Total_Loss <= Loss_Min:
        print("Epoch {:5d} | Loss = {:.5f}".format(epoch+1, Total_Loss))
        Loss_Min = Total_Loss
        No_Improve_Steps = 0
        torch.save(model, "../Model/Model_v0.0")
        print("%5d, %.5f"%(epoch+1, Total_Loss), file=lossfile)
        lossfile.flush()
        Plot_Tsne(D1.cpu(), D2_remove_batch.cpu().detach(), "../Figures/Figures_seed{}/Figures_TSNE_{}.png".format(seed,str(epoch+1)))
    else:
        No_Improve_Steps += 1
    
    if No_Improve_Steps == Max_No_Improve_Steps:
        print("Stop because no improve of loss for {} steps".format(Max_No_Improve_Steps))
        break
