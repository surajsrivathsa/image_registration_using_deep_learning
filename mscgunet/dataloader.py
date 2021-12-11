import torch.utils.data as Data
import numpy as np
import nibabel as nb

def load_4D(name):
    model_np = np.zeros(shape=(128, 128, 128))
    X_nb = nb.load(name)
    X_np = X_nb.dataobj
    model_np = np.reshape(X_np, (1, )+ X_np.shape)
    return model_np

def imgnorm(N_I,index1=0.0001,index2=0.0001):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1*len(I_sort))]
    I_max = I_sort[-int(index2*len(I_sort))]
    N_I =1.0*(N_I-I_min)/(I_max-I_min)
    N_I[N_I>1.0]=1.0
    N_I[N_I<0.0]=0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2


class Dataset(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, t1_filenames, t2_filenames, iterations=1, norm=True):
        'Initialization'
        self.t1_filenames = t1_filenames
        self.t2_filenames = t2_filenames
        self.norm = norm
        self.iterations = iterations

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.t1_filenames) * self.iterations

    def __getitem__(self, idx):
        'Generates one sample of data'

        img_A = load_4D(self.t1_filenames[idx])
        img_B = load_4D(self.t2_filenames[idx])

        if self.norm:
            full_img_A = imgnorm(img_A)
            full_img_B = imgnorm(img_B)
            return full_img_A, full_img_B
        else:
            return img_A, img_B

