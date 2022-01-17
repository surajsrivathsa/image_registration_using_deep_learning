import os, sys, nibabel as nb, glob, torch, losses, layers, model, dataloader, argparse
import losses, model, dataloader, layers


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--fixed", required=True, help="Path to fixed image directory")
    ap.add_argument("-m", "--moving", required=True, help="Path to moving image directory")
    ap.add_argument("-w", "--warped", required=True, help="Path to warped image directory")
    args = vars(ap.parse_args())
    data_path_fixed = args["fixed"]
    data_path_moving = args["moving"]
    data_path_warped = args["warped"]
    # os.environ["WANDB_API_KEY"] = "4d55d2ea028525eadeb537494a02bf9ce8ead8f3"
    # wandb.login()

    file_names_fixed = sorted(glob.glob(os.path.join(data_path_fixed, "*.nii.gz")))
    file_names_moving = sorted(glob.glob(os.path.join(data_path_moving, "*.nii.gz")))
    data_loader_obj = dataloader.direct_optimization_dataset(file_names_fixed, file_names_moving)
    fixed_img_tnsr_lst, moving_img_tnsr_lst, deformation_field_tnsr_lst = data_loader_obj.generate_dataset()
    adadelta_learning_rate = 1e-3
    adagrad_learning_rate = 1e-3
    adam_learning_rate = 1e-3
    adamw_learning_rate = 1e-3
    rmsprop_learning_rate = 1e-3
    sgd_learning_rate = 1e-3

    optimizer_dict = {'optimizer_adadelta': [ torch.optim.Adadelta([deformation_field], lr=adadelta_learning_rate) for deformation_field in deformation_field_tnsr_lst],
                  'optimizer_adagrad': [ torch.optim.Adagrad([deformation_field], lr=adagrad_learning_rate) for deformation_field in deformation_field_tnsr_lst],
                  'optimizer_adam': [ torch.optim.Adam([deformation_field], lr=adam_learning_rate) for deformation_field in deformation_field_tnsr_lst],
                  'optimizer_adamw': [ torch.optim.Adam([deformation_field], lr=adamw_learning_rate) for deformation_field in deformation_field_tnsr_lst],
                  'optimizer_rmsprop': [ torch.optim.RMSprop([deformation_field], lr=rmsprop_learning_rate) for deformation_field in deformation_field_tnsr_lst],
                  'optimizer_sgd': [ torch.optim.SGD([deformation_field], lr=sgd_learning_rate) for deformation_field in deformation_field_tnsr_lst]
                  }

    scheduler_lst = [torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) for optimizer in optimizer_dict['optimizer_rmsprop']]

    spatial_transformer_deformable = layers.SpatialTransformer(size=(128, 128, 128), is_affine=False, affine_image_size =  (1, 1, 128, 128, 128)).to("cuda")
    for param in spatial_transformer_deformable.parameters():
        param.requires_grad = False
        param.volatile=True

    direct_optimization_model = model.DirectOptimizationModel(scheduler_lst=scheduler_lst, deformation_field_tnsr_lst = deformation_field_tnsr_lst, 
                                fixed_img_tnsr_lst = fixed_img_tnsr_lst, moving_img_tnsr_lst = moving_img_tnsr_lst, optimizer_dict=optimizer_dict['optimizer_rmsprop'],
                                data_path_lst_fixed=file_names_fixed, data_path_lst_moving=file_names_moving, data_path_warped=data_path_warped, 
                                spatial_transformer_deformable=spatial_transformer_deformable)
    

    losses_dict = direct_optimization_model.direct_optimization_training()
    losses_dict = model.direct_optimization_training(epochs = 1001, decay_flag=False)