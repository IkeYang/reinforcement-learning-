#Author:ike yang
import sys
sys.path.append(r"c:\users\user\anaconda3\lib\site-packages")
import torch
import warnings
warnings.filterwarnings('ignore')
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim as optim
import torch.nn as nn
from model import LinearModel
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np

def MAPELoss(output, target):
  return torch.mean(torch.abs((target - output) / (target+1e-12)))


def trainLinear(maxiter=20,wp=(6*5,6*1),lamuda=0.00,printOut=True,k0=10,t=4):
    class SCADADataset(Dataset):
        # +jf33N train together
        def __init__(self, name):
            filename = 'dataGW'+name
            with open(filename, 'rb') as f:
                self.dataH,self.dataT= pickle.load(f)
            kMeans=KMeans(k0).fit(self.dataH)
            self.labels=kMeans.labels_

        def __len__(self):
            return self.dataH.shape[0]

        def __getitem__(self, idx):
            x = np.copy(self.dataH[idx,:])
            x = torch.from_numpy(x).float()
            lb=self.labels[idx]
            y = np.copy(self.dataT[idx])
            # y = torch.from_numpy(y).float()
            return x,y,lb
    windL, predL = wp
    (inputD,outD)=(windL,predL)
    batch_size = 1024

    lr = 2e-4
    weight_decay=0.0000

    print(' lr: ',lr,' weight_decay: ',weight_decay,' windL: ',windL,' predL: ',predL,
          ' batch_size: ',batch_size,' inputD: ',inputD,' outD: ',outD,' lamuda:',lamuda)
    epochs = maxiter
    start_epoch = 0
    loadModel = False

    outf = r'C:\YANG Luoxiao\Model\WindSpeed'


    model=LinearModel(inputD,outD).to(device)
    optimizer = optim.Adam(list(model.parameters()), lr=lr,weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=20, verbose=True)
    minloss = 10

    # if loadModel:
    #     checkpoint = torch.load('%s/%s%d.pth' % (outf, "LSTMMutiTS4Best", num))  # largeNew5 Large5
    #     model.load_state_dict(checkpoint['model'])
    #     # D.load_state_dict(checkpoint['D'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     # optimizerD.load_state_dict(checkpoint['optimizerD'])
    #     start_epoch = num
    scadaTrainDataset = SCADADataset( name='Train')
    dataloader = torch.utils.data.DataLoader(scadaTrainDataset, batch_size=batch_size,
                                             shuffle=True, num_workers=int(0))
    scadaValDataset = SCADADataset(name='Val')
    dataloaderVAl = torch.utils.data.DataLoader(scadaValDataset, batch_size=1024,
                                                shuffle=True, num_workers=int(0))
    lossTrain=np.zeros(([1,1]))
    lossVal=np.zeros(([1,1]))
    acc=np.zeros(([1,1]))
    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()
        ckD = {}
        for kk in range(k0):
            ckD[kk]=0
        lossTotal=0
        for i, (x,y,lb) in enumerate(dataloader):

            loss1=0
            loss2=0
            loss3=0
            optimizer.zero_grad()
            # y = y.to(torch.long)
            x = x.to(device).view(x.shape[0],-1)
            y = y.to(device=device, dtype=torch.int64).view(-1)
            gh=model(x)

            for kk in range(k0):
                ghk=gh[lb==kk,:]

                if ghk.shape[0]==0:
                    ckD[kk] = torch.zeros((windL)).to(device)
                    # k=k0-1
                    continue

                ckD[kk]=torch.sum(ghk,dim=0)/(ghk.shape[0])/(i+1)
                loss1+=F.mse_loss(ghk,ckD[kk])
            for kkk in range(k0):
                for kkkk in range(kkk,k0):

                        loss2+=torch.sum((ckD[kkk]-ckD[kkkk])**2)

            for tt in range(t):

                ght=gh[y==tt,:]
                if ght.shape[0]<2:
                    continue
                loss3+=torch.sum((ght[0,:]-ght[1,:])**2)
            # y=y.long()
            # l1loss=l1Norm(model)
            loss1/=k0
            loss2/=(k0*(k0+1)/2)
            loss3/=t

            # loss=F.mse_loss(tgt_y* 25.55 + 0.4, tgtpred* 25.55 + 0.4)
            loss=loss1+loss2+loss3
            loss.backward()
            lossTrain=np.vstack((lossTrain,loss.detach().cpu().numpy().reshape((-1,1))))
            optimizer.step()

            model.eval()
            trans = np.zeros((k0, t))
            gh = model(torch.from_numpy(scadaTrainDataset.dataH).float().to(device)).detach().cpu().numpy()
            kMeans = KMeans(k0).fit(gh)
            scadaTrainDataset.labels = kMeans.labels_
            for b in range(k0):
                tk0 = scadaTrainDataset.dataT[scadaTrainDataset.labels == b]
                for bb in range(t):
                    trans[b, bb] = np.sum(tk0 == bb) / len(tk0)
            # print(trans)
            classP = np.argmax(trans, axis=1)
            c = 0

            accucry = 0
            with torch.no_grad():
                for p, (x, y,lb) in enumerate(dataloaderVAl):
                    # if p>10:
                    #     break
                    c += 1
                    x = x.numpy()
                    labx = kMeans.predict(x)
                    pry = classP[labx]
                    y = y.view(-1).numpy()

                    # predict=torch.argmax(ypred,dim=1)
                    accucry += np.sum(pry == y)
                lengA = len(scadaValDataset)
                # accucry = accucry.cpu().numpy()
                accucry = accucry / lengA
                # lossVal = np.vstack((lossVal, (loss/c).cpu().numpy().reshape((-1, 1))))
                acc = np.vstack((acc, accucry.reshape((-1, 1))))
            model.train()
            #
            # break
            if printOut:
                if (i) % 20 == 0:
                    print('[%d/%d][%d/%d]\tLoss: %.4f\t '
                          % (epoch, start_epoch + epochs, i, len(dataloader), loss))

        # model.eval()
        # trans = np.zeros((k0, t))
        # gh = model(torch.from_numpy(scadaTrainDataset.dataH).float().to(device)).detach().cpu().numpy()
        # kMeans = KMeans(k0).fit(gh)
        # scadaTrainDataset.labels = kMeans.labels_
        # for b in range(k0):
        #     tk0 = scadaTrainDataset.dataT[scadaTrainDataset.labels == b]
        #     for bb in range(t):
        #         trans[b, bb] = np.sum(tk0 == bb) / len(tk0)
        # print(trans)
        # classP=np.argmax(trans,axis=1)
        # c=0
        # loss = 0
        # accucry=0
               # with torch.no_grad():
        #     for l, (x,y,lb) in enumerate(dataloaderVAl):
        #         c += 1
        #         x = x.numpy()
        #         labx=kMeans.predict(x)
        #         pry=classP[labx]
        #         y = y.view(-1).numpy()
        #
        #         # predict=torch.argmax(ypred,dim=1)
        #         accucry+=np.sum(pry==y)
        #     lengA=len(scadaValDataset)
        #     # accucry=accucry.cpu().numpy()
        #     accucry=accucry/lengA
        #     if printOut:
        #         print('VAL loss= ', loss / c, '  VAL accucry ', accucry)
        # #
        # scheduler.step(loss / c)
        # if minloss > (loss / c):
        #     state = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),
        #              'epoch': epoch}
        #     if lamuda==0:
        #
        #         torch.save(state, '%s/TW10RLLinearWT%d.pth' % (outf, int(predL / 6)))
        #     else:
        #
        #         torch.save(state, '%s/RLlassoWT%d.pth' % (outf, int(predL / 6)))
        #     minloss = loss / c
        #     # minmapeloss=lossm/ c
        #     if printOut:
        #         print('bestaccucry:  ', accucry)
        # k0=k

        # print(k)
    return lossTrain[1:,:],lossVal[1:,:],acc[1:,:]

# linear

# print('Linear')
lossT,lossV,acc=trainLinear(maxiter=10,wp=(5*6,5*6),lamuda=0.00,printOut=True,t=25,k0=40)
with open('TradGWRes','wb') as f:
    pickle.dump((lossT,lossV,acc),f)
with open('TradGWRes','rb') as f:
    lossT,lossV,acc=pickle.load(f)
import matplotlib.pyplot as plt
f, axs = plt.subplots(2, 1)
# axs[0].set_xlabel('MiniBatch')
axs[0].set_ylabel('Loss')
axs[0].plot(lossT,label='Train')
# axs[0].plot(lossV,label='Validation')
# axs[0].legend()
axs[1].set_xlabel('MiniBatch')
axs[1].set_ylabel('Accuracy')
axs[1].plot(acc)
plt.show()
f.savefig('TradGWRes.png', dpi=600, bbox_inches='tight')































