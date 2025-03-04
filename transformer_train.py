import argparse

import os, datetime

import torch
import h5py
import numpy as np
import scipy.io

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from transformer_UNet2 import TransformerUNet as Wave
#from  simple_transformer_UNet import SimpleTransformerUNet as Wave
#from config_torchwavenet import ModelConfig
from config_transformer import ModelConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NoisySeqDataset_Customized(Dataset):

    def __init__(self, file_name1, file_name2, split=None):



        self.file_name1 = file_name1

        self.file_name2 = file_name2

        self.split = split



        file1 = h5py.File(self.file_name1, 'r')

        original_Seq = file1['original_Seq'][()].view(complex)

        original_Seq = np.transpose(original_Seq, (1, 0))

        original_Seq = np.expand_dims(original_Seq, axis=1)

        print('original_Seq.shape=', original_Seq.shape)



        file2 = h5py.File(self.file_name2, 'r')

        noisy_Seq = file2['noisy_Seq'][()].view(complex)

        noisy_Seq = np.transpose(noisy_Seq, (1, 0))

        noisy_Seq = np.expand_dims(noisy_Seq, axis=1)

        print('noisy_Seq.shape=', noisy_Seq.shape)



        indices = np.arange(original_Seq.shape[0])



        self.noisy_Seq = np.concatenate((np.real(noisy_Seq), np.imag(noisy_Seq)), axis=1)

        self.original_Seq = np.concatenate((np.real(original_Seq), np.imag(original_Seq)), axis=1)





        if (self.split == 'train'):

            self.indices = indices[256 * 4:256 * 13 * 3]

            # self.indices = indices[256 * 10:256 * 10 * 3]

        elif (self.split == 'val'):

            self.indices = indices[:256 * 1]

        else:

            self.indices = indices[256 * 1 : 256 * 4]



    def __getitem__(self, index):

        input_noisy = self.noisy_Seq[self.indices[index], :, :]

        input_original = self.original_Seq[self.indices[index], :, :]

        return torch.FloatTensor(input_noisy), torch.FloatTensor(input_original)



    def __len__(self):

        return len(self.indices)



# params

parser = argparse.ArgumentParser()



# data paths
parser.add_argument('--data_root', required=True, help='path to file list of h5 train data')
parser.add_argument('--logging_root', type=str, default='/media/staging/deep_sfm/',
                    required=False, help='path to file list of h5 train data')
# train params

parser.add_argument('--train_test', type=str, required=True, help='path to file list of h5 train data')
parser.add_argument('--experiment_name', type=str, default='', help='path to file list of h5 train data')
parser.add_argument('--checkpoint', type=str, default=None, help='path to file list of h5 train data')
parser.add_argument('--max_epoch', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--sigma', type=float, default=0.05, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate, default=0.002')
parser.add_argument('--batch_size', type=int, default=8, help='start epoch')
parser.add_argument('--reg_weight', type=int, default=0., help='start epoch')
parser.add_argument('--fp16', action='store_true', help='Enable mixed precision training')


opt = parser.parse_args()
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





def params_to_filename(params):

    params_to_skip = ['batch_size', 'max_epoch', 'train_test']

    fname = ''

    for key, value in vars(params).items():

        if key in params_to_skip:

            continue

        if key == 'checkpoint' or key == 'data_root' or key == 'logging_root':

            if value is not None:

                value = os.path.basename(os.path.normpath(value))



        fname += "%s_%s_" % (key, value)

    return fname





def train1(model, dataset, val_dataset):

    dataloader = DataLoader(dataset, batch_size=opt.batch_size)



    if opt.checkpoint is not None:

        model.load_state_dict(torch.load(opt.checkpoint))



    model.train()

    model.to(device)



    # directory structure: month_day/

    dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),

                            datetime.datetime.now().strftime('%H-%M-%S_')) + params_to_filename(opt)



    log_dir = os.path.join(opt.logging_root, 'logs', dir_name)

    run_dir = os.path.join(opt.logging_root, 'runs', dir_name)

    if not os.path.exists(log_dir):

        os.makedirs(log_dir)

    if not os.path.exists(run_dir):

        os.makedirs(run_dir)



    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)



    #writer = SummaryWriter(run_dir)

    iter = 0



    #writer.add_scalar("learning_rate", opt.lr, 0)



    print('Beginning training...')

    for epoch in range(opt.max_epoch):

        for model_input, ground_truth in dataloader:

            ground_truth = ground_truth.to(device)

            model_input = model_input.to(device)



            model_outputs = model(model_input)

            #model.write_updates(writer, model_outputs, ground_truth_normalized, model_input_normalized, iter)



            optimizer.zero_grad()



            loss = model.get_loss(model_outputs, ground_truth)



            loss.backward()

            optimizer.step()



            print("Iter %07d   Epoch %03d   loss %0.10f" %

                  (iter, epoch, loss))



            #writer.add_scalar("scaled_regularization_loss", reg_loss * opt.reg_weight, iter)

            #writer.add_scalar("distortion_loss", dist_loss, iter)

            test(model, val_dataset)

            if not iter:

                # Save parameters used into the log directory.

                with open(os.path.join(log_dir, "params.txt"), "w") as out_file:

                    out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))



            iter += 1

            if iter % 1000 == 0:

                torch.save(model.state_dict(), os.path.join(log_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter)))



        val_loss = test(model, val_dataset)
        #test(model, test_dataset)

        scheduler.step(val_loss.item())



    torch.save(model.state_dict(), os.path.join(log_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter)))



def test1(model, dataset):

    dataloader = DataLoader(dataset, batch_size=opt.batch_size)
    if opt.checkpoint is not None:
        model.load_state_dict(torch.load(opt.checkpoint))

    model.eval()
    model.to(device)


    total_loss_arr = []

    for model_input, ground_truth in dataloader:

        ground_truth = ground_truth.to(device)

        model_input = model_input.to(device)
        model_outputs = model(model_input)
        loss = model.get_loss(model_outputs, ground_truth)
        total_loss_arr.append(torch.mean(loss).item())
        
        #print(f'Instantaneous testing average loss: {np.mean(total_loss_arr):.10f}')
        
           

        #scipy.io.savemat('hardware_denoised_images.mat', {'denoised': model_outputs.detach().cpu().numpy()})

        #scipy.io.savemat('hardware_noised_images.mat', {'noised': model_input.detach().cpu().numpy()})

        #%scipy.io.savemat('hardware_original_images.mat', {'original': ground_truth.detach().cpu().numpy()})
        
    print(f'testing average loss.{np.mean(total_loss_arr):.10f}')

    return np.mean(total_loss_arr)
def train(model, dataset, val_dataset):
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=4, pin_memory=True)
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    scaler = torch.cuda.amp.GradScaler(enabled=opt.fp16)
    print('Beginning training...')

    for epoch in range(opt.max_epoch):
        for model_input, ground_truth in dataloader:
            model_input, ground_truth = model_input.to(device), ground_truth.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=opt.fp16):
                model_outputs = model(model_input)
                loss = model.get_loss(model_outputs, ground_truth)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"Epoch {epoch} Loss: {loss:.10f}")

        val_loss = test(model, val_dataset)
        scheduler.step(val_loss)

    torch.save(model.state_dict(), os.path.join(opt.logging_root, 'optimized_model.pth'))

def test(model, dataset):
    dataloader = DataLoader(dataset, batch_size=opt.batch_size)
    model.eval()
    model.to(device)

    total_loss_arr = []
    for model_input, ground_truth in dataloader:
        model_input, ground_truth = model_input.to(device), ground_truth.to(device)
        with torch.no_grad():
            model_outputs = model(model_input)
        loss = model.get_loss(model_outputs, ground_truth)
        total_loss_arr.append(loss.item())
    
    return np.mean(total_loss_arr)

def main():
    #dataset = NoisyCIFAR10Dataset(data_root=opt.data_root,
    #                             sigma=opt.sigma,
    #                             train=opt.train_test == 'train')
    origian_Seq_file = "/uufs/chpc.utah.edu/common/home/u1110463/RF_projects/rf_transformer/Datasets/original_Sig1_5120_5.mat"
    noisy_Seq_file = "/uufs/chpc.utah.edu/common/home/u1110463/RF_projects/rf_transformer/Datasets/noisy_Sig1_5120_5.mat"
    origian_Seq_test = "/uufs/chpc.utah.edu/common/home/u1110463/RF_projects/rf_transformer/Datasets/original_RF.mat"
    #noisy_Seq_test = "/uufs/chpc.utah.edu/common/home/u1110463/RF_projects/rf_transformer/Datasets/noisy_RF.mat"
    #test_dataset = NoisySeqDataset_Customized( origian_Seq_test,noisy_Seq_test,split='val')	
    train_dataset = NoisySeqDataset_Customized(origian_Seq_file, noisy_Seq_file,split = 'train' )
    test_dataset = NoisySeqDataset_Customized(origian_Seq_file, noisy_Seq_file, split='val')
  
    cfg = ModelConfig(
        input_channels=2,  # Keep input as two channels (Real + Imaginary)
        hidden_dim=96,  # Match initial hidden dimension
        max_hidden_dim=768,  # Reduce max hidden dimension for less computation
        encoder_depth=6,  # Shallower encoder
        bottleneck_depth=12,  # Fewer Transformer layers
        decoder_depth=6,  # Shallower decoder
        kernel_size=5,  # Slightly larger kernel for better feature extraction
        stride=2,  # Maintain stride for downsampling
        attention_heads=8,  # Reduce attention heads
        model_dim=512,  # Align model_dim with bottleneck
        inner_dim=2048,  # Reduce feedforward layer size
    )


    cfg1 = ModelConfig(
        input_channels=2,  # Keep input as two channels (Real + Imaginary)
        hidden_dim=96,  # Match the initial hidden dimension with paper
        max_hidden_dim=1024,  # Maintain the max hidden cap
        encoder_depth=8,  # Adjust encoder depth to match UNet transformer
        bottleneck_depth=24,  # Transformer depth
        decoder_depth=8,  # Match encoder depth
        kernel_size=3,
        stride=2,
        attention_heads=16,  # Maintain self-attention heads
        model_dim=1024,  # Align model_dim with bottleneck
        inner_dim=4096,  # Ensure correct feedforward network dimension
    )





    model = Wave(cfg).quantize_model()
    #model = Wave(cfg)
    #model = Wave(input_channels=2)
    train(model, train_dataset, test_dataset)
   # test(model, test_dataset)

if __name__ == '__main__':
    main()