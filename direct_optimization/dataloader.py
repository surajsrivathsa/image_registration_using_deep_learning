import torch.utils.data as Data
import numpy as np
import nibabel as nb
import torch

def load_3D( name):
    # model_np = np.zeros(shape=(160, 192, 224))
    # X_nb = nb.load(name)
    # X_np = X_nb.dataobj
    # #print("Oreintation: {}".format(nb.aff2axcodes(X_nb.affine)))
    # model_np[:, :, :] = X_np[42:202, 32:224, 16:240]
    # #model_np = np.reshape(model_np, (1,)+ model_np.shape)
    # return model_np
    resamplng_shape = (128, 128, 128)
    

    X_nb = nb.load(name)
    X_np = X_nb.dataobj
    least_intensity = np.min(X_np)
    model_np = np.full(shape=resamplng_shape, fill_value=least_intensity)

    #X_np = imgnorm(X_np_old)
    x_dim, y_dim, z_dim = X_np.shape
    x_ltail = (resamplng_shape[0] - x_dim)//2 
    y_ltail = (resamplng_shape[1] - y_dim)//2
    z_ltail = (resamplng_shape[2] - z_dim)//2

    x_rtail = resamplng_shape[0] - x_ltail - 1
    y_rtail = resamplng_shape[1] - y_ltail - 1
    z_rtail = resamplng_shape[2] - z_ltail - 1
    #print("Oreintation: {}".format(nb.aff2axcodes(X_nb.affine)))
    #model_np[:, :, :] = X_np[42:202, 32:224, 16:240]
    model_np[x_ltail:x_rtail, y_ltail:y_rtail, z_ltail:z_rtail] = X_np[:, :, :]
    model_np = np.reshape(model_np, (1,)+ model_np.shape)
    model_np = np.reshape(model_np, (1,)+ model_np.shape)
    #myimg = imgnorm(model_np)
    return model_np

def load_4D(name):
    # (256, 256, 256)
    model_np = np.zeros(shape=(128, 128, 128))
    X_nb = nb.load(name)
    
    X_np = X_nb.dataobj
    #print("Oreintation: {}".format(nb.aff2axcodes(X_nb.affine)))
    #model_np[:, :, 0:X_np.shape[2]] = X_np[0:128, 0:128, :]
    #model_np = np.reshape(model_np, (1,)+ model_np.shape)
    model_np = np.reshape(X_np, (1,)+ X_np.shape)
    model_np = np.reshape(model_np, (1,)+ model_np.shape)
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


def Norm_Zscore( img):
    img= (img-np.mean(img))/np.std(img) 
    return img


def run3functions( fp):
    myimg = load_4D(fp)
    myimg2 = imgnorm(myimg)
    return myimg2


class direct_optimization_dataset:
    def __init__(self, data_path_lst_fixed, data_path_lst_moving):
        self.deformation_field_tnsr_lst = [ ]
        self.fixed_img_tnsr_lst = []
        self.moving_img_tnsr_lst = []
        self.data_path_lst_moving = data_path_lst_moving
        self.data_path_lst_fixed = data_path_lst_fixed
        return
    
    def generate_dataset(self):  
        for i in range(len(self.data_path_lst_fixed)):
            deformation_field_tnsr = torch.randn(size = (1,3,128,128,128)).to("cuda") * 0.00001
            deformation_field_tnsr.requires_grad = True
            self.deformation_field_tnsr_lst.append(deformation_field_tnsr)

            fixed_img_tnsr = torch.from_numpy(run3functions(self.data_path_lst_fixed[i])).to("cuda")
            fixed_img_tnsr.requires_grad = False
            self.fixed_img_tnsr_lst.append(fixed_img_tnsr)

            moving_img_tnsr = torch.from_numpy(run3functions(self.data_path_lst_moving[i])).to("cuda")
            moving_img_tnsr.requires_grad = False
            self.moving_img_tnsr_lst.append(moving_img_tnsr)


        return (self.fixed_img_tnsr_lst, self.moving_img_tnsr_lst, self.deformation_field_tnsr_lst)
