##### base class for models
class regression_model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.loss_func = nn.MSELoss()
        self.lr = 0.0001
        self.patience = 5

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss =self.loss_func(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        return loss

    def configure_optimizers(self):
        self.opt=torch.optim.Adam(self.parameters(), lr=self.lr)
        self.reduce_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt,
                                                                     mode = 'min',
                                                                     factor = 0.2,
                                                                    patience = self.patience,
                                                                    min_lr = 1e-7,
                                                                    verbose = True)
        schedulers =  {'scheduler':self.reduce_lr,'monitor':"val_loss",}
        return [self.opt],schedulers
    
class dilated_residual(pl.LightningModule):
    def __init__(self,filter_num,kernel_size,dilation_rate,dropout):
        super().__init__()
        layers = []
        layers.append(nn.Conv1d(filter_num,filter_num,kernel_size,padding='same'))
        layers.append(nn.BatchNorm1d(filter_num))
        for i in range(0,len(dilation_rate)):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Conv1d(filter_num,filter_num,kernel_size,
                                    padding = 'same',dilation = dilation_rate[i]))
            layers.append(nn.BatchNorm1d(filter_num))
        self.block=nn.Sequential(*layers)
        self.output_act = nn.ReLU()

    def forward(self,x):
        out = self.block(x)
        residual = torch.add(out,x)
        output = self.output_act(residual)
        return output

##### MPRA models #####
class MPRA_model(regression_model):
    def __init__(self,filter_num,kernel_size,residual_num,exp_num,input_len=230,lr = 0.001,patience=5):
        super().__init__()
        #hyper parameters
        self.lr = lr
        self.patience = patience
        
        #first layer filter + batchnorm
        self.first_conv = nn.Conv1d(4,filter_num,kernel_size,padding='same')
        self.batch1 = nn.BatchNorm1d(filter_num)
        self.maxpool = nn.MaxPool1d(8)

        #diltaion layers
        dilation_rate = []
        for i in range(0,residual_num):
            dilation_rate.append(2**i)
        residual_block = []
        residual_block.append(dilated_residual(filter_num,kernel_size,dilation_rate,0.1))
        residual_block.append(nn.MaxPool1d(4))
        self.residual = nn.Sequential(*residual_block)
        self.flat = nn.Flatten()
        out_len = int(input_len/8/4) * filter_num
        self.linear = nn.Linear(out_len,filter_num)
        self.batch2 = nn.BatchNorm1d(filter_num)
        self.last_act = nn.ReLU()
        self.last_dropout = nn.Dropout(0.3)

        #dense layers
        self.dense=nn.Linear(filter_num,128)
        self.dense2 = nn.Linear(128,exp_num)

        self.loss_func = nn.MSELoss()

    def forward(self,x):
        out = self.first_conv(x)
        out = self.batch1(out)
        out = torch.exp(out)
        out = self.maxpool(out)
        out = self.residual(out)
        out = self.flat(out)
        out = self.linear(out)
        out = self.batch2(out)
        out = self.last_act(out)
        out = self.last_dropout(out)
        out = self.dense(out)
        output = self.dense2(out)
        return torch.squeeze(output)

##### Data class #####
class onehot_dataset(Dataset):
    def __init__(self,h5_path):
        self.h5_file = h5py.File(h5_path, "r")

    def __len__(self):
        return len(self.h5_file['onehot'])

    def __getitem__(self,index):
        inputs = self.h5_file['onehot'][index].astype(np.float32)
        targets = self.h5_file['target'][index].astype(np.float32)
        return (inputs, targets)
 