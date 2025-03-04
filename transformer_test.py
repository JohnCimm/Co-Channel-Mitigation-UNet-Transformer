import argparse
import os, datetime
import torch
import h5py
import numpy as np
import scipy.io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformer_UNet import TransformerUNet as Wave
from config_transformer import ModelConfig

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
            self.indices = indices

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
parser.add_argument('--max_epoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--sigma', type=float, default=0.05, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.001')
parser.add_argument('--batch_size', type=int, default=256, help='start epoch')

parser.add_argument('--reg_weight', type=int, default=0., help='start epoch')

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


def test(model, dataset):
    dataloader = DataLoader(dataset, batch_size=opt.batch_size)

    if opt.checkpoint is not None:
        model.load_state_dict(torch.load(opt.checkpoint))

    model.eval()
    model.to(device)
    
    print('Beginning testing...')
    total_loss_arr = []
    all_inputs = []
    all_outputs = []
    all_ground_truths = []

    for model_input, ground_truth in dataloader:
        ground_truth = ground_truth.to(device, non_blocking=True)
        model_input = model_input.to(device, non_blocking=True)

        with torch.no_grad():
            model_outputs = model(model_input)

        loss = model.get_loss(model_outputs, ground_truth)

        total_loss_arr.append(torch.mean(loss).item())
        all_inputs.append(model_input.cpu().detach().numpy())
        all_outputs.append(model_outputs.cpu().detach().numpy())
        all_ground_truths.append(ground_truth.cpu().detach().numpy())
        #scipy.io.savemat('testwave_denoised_images.mat', {'denoised': model_outputs.cpu().detach().numpy()})
        #scipy.io.savemat('testwave_noised_images.mat', {'noised': model_input.cpu().detach().numpy()})
        #scipy.io.savemat('testwave_original_images.mat', {'original': ground_truth.cpu().detach().numpy()})
    
    # Convert lists to arrays
    all_inputs = np.concatenate(all_inputs, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_ground_truths = np.concatenate(all_ground_truths, axis=0)

    # Save to a single .mat file
    scipy.io.savemat('test_transformer_results_hardware_02042025Batch4_RNG15_ManualVsMatlabCFO.mat', {
        'noised': all_inputs,
        'denoised': all_outputs,
        'original': all_ground_truths
    })
    
    print(f'testing average loss: {np.mean(total_loss_arr):.10f}')


def main():
    #origian_Seq_file = "/uufs/chpc.utah.edu/common/home/u01110463/RF_projects/rf_transformer/Datasets/original_test_hardware_snr5_70images.mat"
    #noisy_Seq_file = "/uufs/chpc.utah.edu/common/home/u01110463/RF_projects/rf_transformer/Datasets/noisy_test_hardware_snr5_70images.mat"
    origian_Seq_file = "Datasets/original_test_hardware_3.mat"
    noisy_Seq_file = "Datasets/noisy_test_hardware_3.mat"

    test_dataset = NoisySeqDataset_Customized(origian_Seq_file, noisy_Seq_file)

    
    
    cfg = ModelConfig(
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
    
    cfg1 = ModelConfig(
        input_channels=2,  # Keep input as two channels (Real + Imaginary)
        hidden_dim=96,  # Match initial hidden dimension
        max_hidden_dim=768,  # Reduce max hidden dimension for less computation
        encoder_depth=6,  # Shallower encoder
        bottleneck_depth=12,  # Fewer Transformer layers
        decoder_depth=6,  # Shallower decoder
        kernel_size=5,  # Slightly larger kernel for better feature extraction
        stride=2,  # Maintain stride for downsampling
        attention_heads=8,  # Reduce attention heads
        inner_dim=2048,  # Reduce feedforward layer size
    )

    model = Wave(cfg)
    test(model, test_dataset)


if __name__ == '__main__':
    main()