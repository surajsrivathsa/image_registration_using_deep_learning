import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
from layers import ResizeTransform, VecInt, SpatialTransformer
import os, glob, sys, nibabel as nb
from losses import Grad, NormalizedCrossCorrelation, MutualInformation
import losses

class DirectOptimizationModel:
    
    def __init__(self, scheduler_lst, deformation_field_tnsr_lst, optimizer_dict, moving_img_tnsr_lst, fixed_img_tnsr_lst, 
                data_path_lst_fixed, data_path_lst_moving, data_path_warped, spatial_transformer_deformable ):
        self.dim = 3
        self.int_downsize = 2
        self.down_shape = [int(self.dim/ self.int_downsize) for dim in (128, 128, 128)]
        self.resize = ResizeTransform(2, 3).to("cuda")
        self.fullsize = ResizeTransform(0.5, 3).to("cuda")
        self.integrate = VecInt(self.down_shape, 7).to("cuda")
        self.smoothness_loss = Grad(penalty='l2')
        self.similarity_loss = NormalizedCrossCorrelation().to("cuda")
        self.scheduler_lst = scheduler_lst
        self.deformation_field_tnsr_lst = deformation_field_tnsr_lst
        self.optimizer_dict = optimizer_dict
        self.moving_img_tnsr_lst = moving_img_tnsr_lst 
        self.fixed_img_tnsr_lst = fixed_img_tnsr_lst
        self.data_path_lst_moving = data_path_lst_moving
        self.data_path_lst_fixed = data_path_lst_fixed
        self.data_path_warped = data_path_warped
        self.spatial_transformer_deformable = spatial_transformer_deformable
        
        return

    
    def optimize_one_image_pair(self, fixed_img_tnsr, moving_img_tnsr, deformation_field, optimizer , scheduler, epochs=1001, decay_flag=False):
        optimized_deformation_field_lst = []
        loss_lst = []
        scheduler = self.scheduler_lst[0]
        # optimizer = self.optimizer_dict['optimizer_adam'][0]
        
        for step in range(epochs):
            
            # Find warped image given moving image from spatial transformer
            pos_flow = self.resize(deformation_field)
            integrated_pos_flow = self.integrate(pos_flow)
            full_flow = self.fullsize(integrated_pos_flow)
            warped_image_tensor = self.spatial_transformer_deformable(moving_img_tnsr.contiguous(), full_flow)
            
            # Find loss between warped image and fixed image
            similarity_loss_ncc = -1.0 * self.similarity_loss(fixed_img_tnsr.contiguous(), warped_image_tensor.contiguous())
            sm_loss =  0.5 * self.smoothness_loss.loss("",deformation_field)
            similarity_loss_mae = 0.1 * losses.mae_loss(fixed_img_tnsr.contiguous(), warped_image_tensor.contiguous())
            # similarity_loss_mae = 0.0
            total_loss = similarity_loss_ncc + sm_loss + similarity_loss_mae

            # Backpropagate loss through network
            optimizer.zero_grad()          
            total_loss.backward()
            optimizer.step()

            # Decay if the falg is true
            if decay_flag:
                scheduler.step()

            # Log losses
            if (step % 500 == 0):
            # loss_lst.append(step, similarity_loss_ncc, sm_loss, similarity_loss_mae, total_loss)
                print("losses at {} are {}, {}, {} and {}".format(step, similarity_loss_ncc, sm_loss, similarity_loss_mae, total_loss))

            # Save deformation field every multiple of 500 epoch that is 500,1000,1500,2000
            if (step % 500 == 0) and (step > 0):
            # deformation_field_detached = deformation_field.detach()
            # optimized_deformation_field_lst.append(deformation_field_detached)
                pass
            
            # delete unused warped image to save memeory
            del warped_image_tensor

        # Save deformation field
        deformation_field_detached = deformation_field.detach()
        optimized_deformation_field_lst.append(deformation_field_detached)

        # clear memory of deformation field
        del deformation_field

        return (optimized_deformation_field_lst, loss_lst) 



    def generate_and_save_warped_images( self, fixed_img_path, moving_img_path, fixed_img_tnsr, moving_img_tnsr, optimized_deformation_field_lst, data_path_warped):
        fixed_image_name = os.path.basename(fixed_img_path).replace('.nii.gz', '')
        moving_image_name = os.path.basename(moving_img_path).replace('.nii.gz', '')
        warped_image_name = fixed_image_name + "|" + moving_image_name
        counter = 1
        for deformation_field in optimized_deformation_field_lst:
            pos_flow = self.resize(deformation_field)
            integrated_pos_flow = self.integrate(pos_flow)
            full_flow = self.fullsize(integrated_pos_flow)
            warped_img_tnsr = self.spatial_transformer_deformable(moving_img_tnsr, full_flow)
            warped_img_np = warped_img_tnsr.to("cpu").numpy()
            warped_img_nb = nb.Nifti1Image(warped_img_np[0,0,:,:,:], np.eye(4))
            nb.save(warped_img_nb, os.path.join(data_path_warped , warped_image_name + "_" + str(counter) + ".nii.gz"))
            counter = counter + 1
        
        return




    def direct_optimization_training(self, epochs = 1001, decay_flag=False):
        losses_dict = {}
        counter = 0

        for fixed_img_path, moving_img_path, fixed_img_tnsr, moving_img_tnsr, deformation_field in zip(self.data_path_lst_fixed, self.data_path_lst_moving, self.fixed_img_tnsr_lst, self.moving_img_tnsr_lst, self.deformation_field_tnsr_lst):
            fixed_image_name = os.path.basename(fixed_img_path).replace(".nii.gz", "")
            moving_image_name = os.path.basename(moving_img_path).replace(".nii.gz", "")
            warped_image_name = fixed_image_name + "|" + moving_image_name

            print("Optimizing {} image pair for {}, {}".format(counter, fixed_image_name, moving_image_name ))

            # obtain a list of deformation field at multiple of 500 epochs and also losses list at every multiple of 100th epoch
            optimized_deformation_field_lst, loss_lst = self.optimize_one_image_pair(fixed_img_tnsr, moving_img_tnsr, deformation_field, 
                                                                                optimizer = self.optimizer_dict[counter], scheduler = self.scheduler_lst[counter], 
                                                                                epochs=epochs, decay_flag=decay_flag)
            
            # saving the warped images for specific epoch deformation fields in the folder path
            self.generate_and_save_warped_images(fixed_img_path, moving_img_path, fixed_img_tnsr, moving_img_tnsr, optimized_deformation_field_lst, data_path_warped=self.data_path_warped)

            losses_dict[warped_image_name] = loss_lst
            print("")
            print(" ============ ================== ============== =============== ============ ================ ================ ")
            print("")
            
            counter = counter + 1

            del optimized_deformation_field_lst
        
        return losses_dict
        

