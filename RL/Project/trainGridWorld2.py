#Author:ike yang
import sys
sys.path.append(r"c:\users\user\anaconda3\lib\site-packages")
import torch
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.optim as optim
import torch.nn as nn
from model import LinearModel
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np

def MAPELoss(output, target):
  return torch.mean(torch.abs((target - output) / (target+1e-12)))


def trainLinear(maxiter=20,wp=(6*5,6*1),lamuda=0.00,printOut=True):

    class SCADADataset(Dataset):
        # +jf33N train together
        def __init__(self, name):
            filename = 'dataGW'+name+'2'
            with open(filename, 'rb') as f:
                self.dataH,self.dataT= pickle.load(f)


        def __len__(self):
            return self.dataH.shape[0]

        def __getitem__(self, idx):
            x = np.copy(self.dataH[idx,:])
            x = torch.from_numpy(x).float()

            y = np.copy(self.dataT[idx])
            y = torch.from_numpy(y).float()
            return x,y
    windL, predL = wp
    (inputD,outD)=(windL,predL)
    batch_size = 64

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
    dataloaderVAl = torch.utils.data.DataLoader(scadaValDataset, batch_size=512,
                                                shuffle=True, num_workers=int(0))

    for epoch in range(start_epoch, start_epoch + epochs):
        model.train()


        for i, (x,y) in enumerate(dataloader):


            optimizer.zero_grad()
            # y = y.to(torch.long)
            x = x.to(device).view(x.shape[0],-1)
            y = y.to(device=device, dtype=torch.int64).view(-1)
            ypred=model(x)
            # y=y.long()
            # l1loss=l1Norm(model)


            # loss=F.mse_loss(tgt_y* 25.55 + 0.4, tgtpred* 25.55 + 0.4)
            loss=F.cross_entropy(ypred, y)
            loss.backward()

            optimizer.step()

            #
            # break
            if printOut:
                if (i) % 2000 == 0:
                    print('[%d/%d][%d/%d]\tLoss: %.4f\t '
                          % (epoch, start_epoch + epochs, i, len(dataloader), loss))

        model.eval()
        c=0
        loss = 0
        accucry=0
        with torch.no_grad():
            for i, (x,y) in enumerate(dataloaderVAl):
                c += 1
                x = x.to(device).view(x.shape[0], -1)
                y = y.to(device=device, dtype=torch.int64).view(-1)
                ypred = model(x)
                loss += F.cross_entropy(ypred, y)
                predict=torch.argmax(ypred,dim=1)
                accucry+=torch.sum(predict==y)
            lengA=len(scadaValDataset)
            accucry=accucry.cpu().numpy()
            accucry=accucry/lengA
            if printOut:
                print('VAL loss= ', loss / c, '  VAL accucry ', accucry)

        scheduler.step(loss / c)
        if minloss > (loss / c):
            state = {'model': model.state_dict(),'optimizer': optimizer.state_dict(),
                     'epoch': epoch}
            if lamuda==0:

                torch.save(state, '%s/GW2RLLinearWT%d.pth' % (outf, int(predL / 6)))
            else:

                torch.save(state, '%s/RLlassoWT%d.pth' % (outf, int(predL / 6)))
            minloss = loss / c
            # minmapeloss=lossm/ c
            if printOut:
                print('bestaccucry:  ', accucry)



# linear

print('Linear')
trainLinear(maxiter=100,wp=(5*6,5*5),lamuda=0.00,printOut=True)



































