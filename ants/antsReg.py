import ants
import nibabel as nib
import numpy as np
import glob
from skimage.transform import resize
import os
import time

data_path_m = "<moving images path>"
data_path_f = "<fixed image path>"
file_names_f = sorted(glob.glob(os.path.join(data_path_f, "*.nii.gz")))
file_names_m = sorted(glob.glob(os.path.join(data_path_m, "*.nii.gz")))
for fixed, moving in zip(file_names_f, file_names_m):
     print(os.path.basename(fixed))
     print(os.path.basename(moving))
     t1_fixed = ants.image_read(fixed)
     t2_moving = ants.image_read(moving)
     mytx = ants.registration(fixed=t1_fixed, moving=t2_moving, type_of_transform='Affine')
     warped_moving = mytx['warpedmovout']
     ants_output = '<result path>'+ os.path.basename(fixed)[:-7] + "_F_" + os.path.basename(moving)[:-7] + "_M.nii.gz"
     ants.image_write(warped_moving, ants_output, ri=False)
