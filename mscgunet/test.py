from dataloader import *
from model import *
from losses import *
from layers import *
import glob
import os
import torch
import time
import argparse

def fullmodel_one_epoch_run(epoch=1):
    example_number = 0
    counter = 0
    for X, Y in training_generator:
        X = X.float().to("cuda")
        Y = Y.float().to("cuda")

        # ============================================ X-Y START =================================================================

        enc_1_xy, enc_2_xy, enc_3_xy, enc_4_xy, enc_5_xy, enc_6_xy = feature_extractor_training(X, Y)
        A_xy, gx_xy, scg_loss_xy, z_hat_xy = scg_training(enc_6_xy)
        B_xy, C_xy, H_xy, W_xy, D_xy = enc_6_xy.size()
        gop_layers1_xy, A_layers1_xy = graph_layers1_training((gx_xy.reshape(B_xy, -1, C_xy), A_xy))
        gop_layers2_xy, A_layers2_xy = graph_layers2_training((gop_layers1_xy, A_layers1_xy))

        gop_layers2_xy = torch.bmm(A_layers2_xy, gop_layers2_xy)
        gop_layers2_xy = gop_layers2_xy + z_hat_xy

        # Upward trajectory
        gx_xy = gop_layers2_xy.reshape(B_xy, 9, 4, 4, 4)
        gx_xy = F.interpolate(gx_xy, (H_xy, W_xy, D_xy), mode='trilinear', align_corners=False)

        # Adding information from feature extractor directly to latent space info, this could provide a path for gradients to move faster
        gx_xy = upsampler1_training(gx_xy, enc_5_xy)  # 8
        gx_xy = upsampler2_training(gx_xy, enc_4_xy)  # 16
        gx_xy = upsampler3_training(gx_xy, enc_3_xy)  # 32
        gx_xy = upsampler4_training(gx_xy, enc_2_xy)  # 64
        gx_xy = upsampler5_training(gx_xy, enc_1_xy)  # 128, 32

        # Concat fixed image to final field before smoothening
        gx_xy = torch.cat((X, gx_xy), 1)

        # Conv decoder last three layers from voxel morph
        gx_xy = conv_decoder1_training(gx_xy)  # 128, 33 --> 16
        gx_xy = conv_decoder2_training(gx_xy)  # 128,16
        dvf_xy = conv_decoder3_training(gx_xy)  # 128,3

        # vector integration for diffeomorphic field

        pos_flow_xy = resize(dvf_xy)
        integrated_pos_flow_xy = integrate(pos_flow_xy)
        full_flow_xy = fullsize(integrated_pos_flow_xy)
        fully_warped_image_xy = stn_deformable(Y, full_flow_xy)

        # ============================================= X-Y END ====================================================================

        # ============================================= Y-X START ==================================================================

        enc_1_yx, enc_2_yx, enc_3_yx, enc_4_yx, enc_5_yx, enc_6_yx = feature_extractor_training(Y, X)
        A_yx, gx_yx, scg_loss_yx, z_hat_yx = scg_training(enc_6_yx)
        B_yx, C_yx, H_yx, W_yx, D_yx = enc_6_yx.size()
        gop_layers1_yx, A_layers1_yx = graph_layers1_training((gx_yx.reshape(B_yx, -1, C_yx), A_yx))
        gop_layers2_yx, A_layers2_yx = graph_layers2_training((gop_layers1_yx, A_layers1_yx))

        gop_layers2_yx = torch.bmm(A_layers2_yx, gop_layers2_yx)
        gop_layers2_yx = gop_layers2_yx + z_hat_yx

        # Upward trajectory
        gx_yx = gop_layers2_yx.reshape(B_yx, 9, 4, 4, 4)
        gx_yx = F.interpolate(gx_yx, (H_yx, W_yx, D_yx), mode='trilinear', align_corners=False)

        # Adding information from feature extractor directly to latent space info, this could provide a path for gradients to move faster
        gx_yx = upsampler1_training(gx_yx, enc_5_yx)  # 8
        gx_yx = upsampler2_training(gx_yx, enc_4_yx)  # 16
        gx_yx = upsampler3_training(gx_yx, enc_3_yx)  # 32
        gx_yx = upsampler4_training(gx_yx, enc_2_yx)  # 64
        gx_yx = upsampler5_training(gx_yx, enc_1_yx)  # 128, 32

        # Concat fixed image to final field before smoothening
        gx_yx = torch.cat((Y, gx_yx), 1)

        # Conv decoder last three layers from voxel morph
        gx_yx = conv_decoder1_training(gx_yx)  # 128, 33 --> 16
        gx_yx = conv_decoder2_training(gx_yx)  # 128,16
        dvf_yx = conv_decoder3_training(gx_yx)  # 128,3

        # Suraj adding vector integration for diffeomorphic field

        pos_flow_yx = resize(dvf_yx)
        integrated_pos_flow_yx = integrate(pos_flow_yx)
        full_flow_yx = fullsize(integrated_pos_flow_yx)
        fully_warped_image_yx = stn_deformable(X, full_flow_yx)

        # ============================================= Y-X END ======================================================================

        fully_warped_image_xy = fully_warped_image_xy.detach().to("cpu").numpy()
        full_warped_nb = nb.Nifti1Image(fully_warped_image_xy[0, 0, :, :, :], np.eye(4))
        nb.save(full_warped_nb,
                args["output"] + os.path.basename(file_names_fixed[counter])[:-7] + "_F_" + os.path.basename(file_names_moving[counter])[:-7] + "_M.nii.gz")

        #dvf_np = dvf_xy.detach().to("cpu").numpy()
        #dvf_nb = nb.Nifti1Image(dvf_np[0, :, :, :, :], np.eye(4))
        #nb.save(dvf_nb, '/nfs1/shashidh/scripts/scgnet/output/' + 'dvf_nb_' + str(counter) + '.nii.gz')

        counter = counter + 1
        del X, Y
        del enc_1_xy, enc_2_xy, enc_3_xy, enc_4_xy, enc_5_xy, enc_6_xy, dvf_xy, integrated_pos_flow_xy, pos_flow_xy
        del A_xy, gx_xy, z_hat_xy, gop_layers1_xy, A_layers1_xy, gop_layers2_xy, A_layers2_xy, fully_warped_image_xy, full_flow_xy
        del enc_1_yx, enc_2_yx, enc_3_yx, enc_4_yx, enc_5_yx, enc_6_yx, dvf_yx, integrated_pos_flow_yx, pos_flow_yx
        del A_yx, gx_yx, z_hat_yx, gop_layers1_yx, A_layers1_yx, gop_layers2_yx, A_layers2_yx, fully_warped_image_yx, full_flow_yx
        torch.cuda.empty_cache()

        example_number = example_number + 1
        #if (counter > 4):
        #    break


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--fixed", required=True, help="Path to fixed image directory")
    ap.add_argument("-m", "--moving", required=True, help="Path to moving image directory")
    ap.add_argument("-pth", "--model", required=True, help="Path to the store the model checkpoints")
    ap.add_argument("-o", "--output", required=True, help="Path to the store the registered results")
    args = vars(ap.parse_args())
    data_path_fixed = args["fixed"]
    data_path_moving = args["moving"]
    model_dir = args["model"]

    # load files
    file_names_fixed = sorted(glob.glob(os.path.join(data_path_fixed, "*.nii.gz")))
    file_names_moving = sorted(glob.glob(os.path.join(data_path_moving, "*.nii.gz")))
    training_generator = Data.DataLoader(Dataset(file_names_fixed, file_names_moving, norm=True), batch_size=1, shuffle=False)


# ========================================= Model Init - START =========================================================

    feature_extractor_training = Feature_Extractor(2,3,16).to("cuda")

    scg_training = SCG_block(in_ch=32, hidden_ch=9, node_size=(4, 4, 4)).to("cuda")

    upsampler1_training = convEncoder(9, 32, 32).to("cuda")
    upsampler2_training = convEncoder(32, 32, 32).to("cuda")
    upsampler3_training = convEncoder(32, 32, 32).to("cuda")
    upsampler4_training = convEncoder(32, 32, 32).to("cuda")
    upsampler5_training = convEncoder(16, 32, 32).to("cuda")

    conv_decoder1_training = convDecoder(33, 16).to("cuda")
    conv_decoder2_training = convDecoder(16, 16).to("cuda")
    conv_decoder3_training = convDecoder(16, 3).to("cuda")

    graph_layers1_training = GCN_Layer(32, 16, bnorm=True, activation=nn.LeakyReLU(0.2), dropout=0.1).to("cuda")
    graph_layers2_training = GCN_Layer(16, 9, bnorm=True, activation=nn.LeakyReLU(0.2), dropout=0.1).to("cuda")

    stn_deformable = SpatialTransformer(size=(128, 128, 128), is_affine=False).to("cuda")

    weight_xavier_init(graph_layers1_training, graph_layers2_training, scg_training)

    for param in stn_deformable.parameters():
        param.requires_grad = False
        param.volatile=True

    # vector integrion to enforce diffeomorphic transform
    dim = 3
    int_downsize = 2
    down_shape = [int(dim / int_downsize) for dim in (128, 128, 128)]
    resize = ResizeTransform(2, 3).to("cuda")
    fullsize = ResizeTransform(0.5, 3).to("cuda")
    integrate = VecInt(down_shape, 7).to("cuda")

    # Loss functions
    similarity_loss = NormalizedCrossCorrelation().to("cuda")
    smoothness_loss = Grad(penalty='l2')

# ========================================= Model Init - END ===========================================================


# ======================================== checkpoint load START =======================================================
    # load previous checkpoints
    checkpoint = torch.load(os.path.join(model_dir, 'scgnet_upsampling_concat_deconv_new_400.pth'),
                            map_location=torch.device('cuda'))

    feature_extractor_training.load_state_dict(checkpoint['feature_extractor_training'])
    scg_training.load_state_dict(checkpoint['scg_training'])

    graph_layers1_training.load_state_dict(checkpoint['graph_layers1_training'])
    graph_layers2_training.load_state_dict(checkpoint['graph_layers2_training'])

    upsampler1_training.load_state_dict(checkpoint["upsampler1_training"])
    upsampler2_training.load_state_dict(checkpoint["upsampler2_training"])
    upsampler3_training.load_state_dict(checkpoint["upsampler3_training"])
    upsampler4_training.load_state_dict(checkpoint["upsampler4_training"])
    upsampler5_training.load_state_dict(checkpoint["upsampler5_training"])

    conv_decoder1_training.load_state_dict(checkpoint["conv_decoder1_training"])
    conv_decoder2_training.load_state_dict(checkpoint["conv_decoder2_training"])
    conv_decoder3_training.load_state_dict(checkpoint["conv_decoder3_training"])
# ======================================== checkpoint load END =========================================================

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
        
    start_time = time.time()
    fullmodel_one_epoch_run()
    end_time = time.time()
    print("Total time taken: {} minutes".format((end_time - start_time) / 60.0))
