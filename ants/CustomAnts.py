# import the necessary packages
import argparse
import os
import SimpleITK as sitk
import ants

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--fixed", required = True, help = "Path to the MRI Image to which we register the moving image ")
ap.add_argument("-m", "--moving", required = True, help = "Path to the MRI image to be mapped to fixed space")
ap.add_argument("-b", "--bias", required = False, help = "Specify type of bias field correction (N3,N4)")
ap.add_argument("-r", "--resample", required = False, help = "Resample to target(fixed) image specified or with specific spacing")
ap.add_argument("-t", "--transform", required = False, help = "Specify the type of transform for registration")
ap.add_argument("-c", "--cmap", required = False, help = "Specify the type of cmap for plot")
args = vars(ap.parse_args())

# Function to show Image details such as dimensions, spacing 
def show_header_info(path):
	header_info = ants.image_header_info(path)
	print("Header Information for: {} \n Number of Dimensions: {} \n Dimensions: {} \n Spacing: {}".format(os.path.basename(path),header_info['nDimensions'],header_info['dimensions'],header_info['spacing']) )
	
# Function to read ANTSImage from file 
def read_image(path):
	return ants.image_read(path)
	
# Function to perform bias field correction
def bias_field_correction(image,correctiontype):
	if correctiontype == 'N3':
		return ants.n3_bias_field_correction(image)
	elif correctiontype == 'N4':
		return ants.n4_bias_field_correction(image)
	else:
		print('incorrect bias field correction type specified. Use N3 or N4')
		
# Function to perform resampling
def resample_image(image,target_image,resample_type):
	if resample_type == 'target':
		return ants.resample_image_to_target(image,target_image)
	else:
		print(resample_type.replace('"',''))
		return ants.resample_image(image,resample_type.replace('"',''),False,1)

# Function to save the plots to current working directory	
def save_plot(image,path,overlay=None,Title=None):
	filename = os.getcwd() + '/plots/{}'.format(os.path.basename(path))+'.png'
	if overlay == None:	
		if args['cmap'] is not None:
			ants.plot(image,filename=filename,cmap=args['cmap'])
		else:
			ants.plot(image,filename=filename)
	else:
		if args['cmap'] is not None:
			ants.plot(image=image,overlay=overlay,filename=filename,title=Title)
		else:
			ants.plot(image=image,overlay=overlay,filename=filename,title=Title,overlay_cmap=args['cmap'])

input_fixed = args['fixed']
input_moving = args['moving']

# Read nifti images using ants
fixed_antsimage = read_image(input_fixed)
moving_antsimage_org = read_image(input_moving)
moving_antsimage = moving_antsimage_org.copy()

# Show Header information of the images
show_header_info(input_fixed)
show_header_info(input_moving)

# Re-orient the moving image to same orientation as the fixed image 
if ants.get_orientation(fixed_antsimage) != ants.get_orientation(moving_antsimage):
	print('Re-orienting moving image to fixed image. {} -> {}'.format(ants.get_orientation(moving_antsimage),ants.get_orientation(fixed_antsimage)) )
	moving_antsimage = ants.reorient_image2(moving_antsimage, (ants.get_orientation(fixed_antsimage)))

# perform bias field correction if bias flag is set	
if args['bias'] is not None: 
	#moving_antsimage = bias_field_correction(moving_antsimage,args["bias"]) # commented as its causing segmentation fault
	print('segmentation fault')

# perform resampling to target image if resample flag is set	
if args['resample'] is not None: 
	moving_antsimage = resample_image(moving_antsimage,fixed_antsimage,args["resample"])
	print('resampled image:', moving_antsimage)
	
if not os.path.exists('plots'):
    os.mkdir('plots') 

#Save fixed and moving images to disk
save_plot(fixed_antsimage,input_fixed)
save_plot(moving_antsimage_org,input_moving)
save_plot(moving_antsimage,'resampled_reoriented')


# perform registration using required transformation
if args['transform'] is not None:
	transforms = ['Translation','Rigid','Similarity','QuickRigid','DenseRigid','BOLDRigid','Affine','AffineFast','BOLDAffine','TRSAA','ElasticSyN','SyN','SyNRA','SyNOnly','SyNCC','SyNabp','SyNBold','SyNBoldAff','SyNAggro','TVMSQ','TVMSQC']
	if args['transform'] not in transforms:
		print('please choose a transform from :',transforms)
	else:
		mytx = ants.registration(fixed=fixed_antsimage , moving=moving_antsimage, type_of_transform=args['transform'])
		warped_moving = mytx['warpedmovout']
else:
	mytx = ants.registration(fixed=fixed_antsimage , moving=moving_antsimage, type_of_transform='Affine' )
	warped_moving = mytx['warpedmovout']
	
	
ants.image_write(warped_moving, 'registered.nii.gz', ri=False)
fixed_antsimage.plot(overlay=moving_antsimage,cmap = 'Reds',overlay_cmap='jet', title='Before Registration',filename=os.getcwd() + "/plots/Before_registration.png")
fixed_antsimage.plot(overlay=warped_moving,cmap = 'Reds',overlay_cmap='jet', title='After Registration',filename=os.getcwd() + "/plots/After_registration.png")
print('============================================Registration complete. Check plots folder for visualiztions.==============================================================')
