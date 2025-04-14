import os
import matlab.engine
import ants
from dipy.io.image import load_nifti, save_nifti
from dipy.denoise.localpca import mppca
from dipy.denoise.gibbs import gibbs_removal
from dipy.align import (affine_registration, center_of_mass, translation, rigid)
from nipype.interfaces import fsl
from time import time


"""

Image processing and T1 mapping for RARE_VTR mouse experiments.


Quantitative Microstructure Imaging Lab
Written by VG 2022


"""

def preprocessing(input_data, input_affine):
    """
    :param input_data:
    :param input_affine:
    :return:
    """
    # start timer
    t = time()

    # Correct for echo misalignment using rigid body registration
    print('Performing motion correction...')
    pipeline = [center_of_mass, translation, rigid]
    output_data = input_data[:, :, :, :]
    for i in range(1, input_data.shape[3]):
        transformed, reg_affine = affine_registration(
            input_data[:, :, :, i],
            input_data[:, :, :, 0],
            moving_affine=input_affine,
            static_affine=input_affine,
            nbins=32,
            metric='MI',
            pipeline=pipeline,
            level_iters=[10000, 1000, 100],
            sigmas=[3.0, 1.0, 0.0],
            factors=[4, 2, 1])
        output_data[:, :, :, i] = transformed
    print('Finished motion correction.')


    # Do Denoising
    print('Denoising with MPPCA...')
    denoised_data = mppca(output_data, patch_radius=1)

    # Correct for Gibbs Ringing
    print('Correcting for Gibbs ringing...')
    corrected_data = gibbs_removal(denoised_data, slice_axis=2)
    print("Completed preprocessing data.")
    print("Elapsed time is: ", -t + time(), "seconds.")
    return corrected_data


def skull_stripping(input_data_path):
    """
    :param input_data_path:
    :return:
    """
    # Start the matlab engine and set up paths
    print('Performing skullstripping with SHERM...')
    eng = matlab.engine.start_matlab()
    sherm_root_dir = '/data/rudko/vgrouza/invivomouse/sherm'
    eng.addpath(sherm_root_dir, nargout=0)

    # Specify SHERM parameters
    descriptor = os.path.join(sherm_root_dir, 'rodent_brain_polar_descriptor.mat')
    animal = 'mouse'
    isotropic = 0

    # Run SHERM
    output_data_path = os.path.join(input_data_path, 'binary_mask.nii.gz')
    input_data_path = os.path.join(input_data_path, 'corrected_data.nii.gz')
    eng.sherm(input_data_path, output_data_path, descriptor, animal, isotropic, nargout=0)
    print('Completed skull stripping.')
    eng.quit()
    return 0


def bias_field_correcting(input_path):

    print('Performing N3 bias field correction...')
    input_data_path = os.path.join(input_path, 'corrected_data.nii.gz')
    input_mask_path = os.path.join(input_path, 'binary_mask.nii.gz')
    data = ants.image_read(input_data_path)
    mask = ants.image_read(input_mask_path)
    bias_field = ants.n3_bias_field_correction2(ants.ndimage_to_list(data)[0]*mask, return_bias_field=True)
    corrected_data_list = list()
    for i in range(0, data.shape[-1]):
        corrected_data_list.append(ants.ndimage_to_list(data)[i] / bias_field)
    corrected_data = ants.list_to_ndimage(data, corrected_data_list)
    ants.image_write(corrected_data, os.path.join(input_path, 'bias_corrected_data.nii.gz'))


def t1map_fitting(input_data_path):
    # Start the matlab engine and set up paths
    print('Performing T1 map fitting with lsqnonneg...')
    eng = matlab.engine.start_matlab()
    mriphys_root_dir = '/data/rudko/vgrouza/invivomouse/mriphysics'
    eng.addpath(mriphys_root_dir, nargout=0)

    # Run T1 map fitting procedure
    mask_data_path = os.path.join(input_data_path, 'binary_mask.nii.gz')
    input_data_path = os.path.join(input_data_path, 'bias_corrected_data.nii.gz')
    eng.rarevtr_t1fit(input_data_path, mask_data_path, nargout=0)
    print('Completed T1 map fitting.')
    eng.quit()
    return 0


def inverse_reg(input_data_path):
    """
    :param input_data_path:
    :return:
    """
    t = time()
    # OBTAIN THE FORWARD TRANSFORM
    flt = fsl.FLIRT(bins=256, cost_func='mutualinfo')
    moving_file_dir = os.path.join(input_data_path, 'masked_first_vtr.nii.gz')
    static_file_dir = os.path.join('/data/rudko/vgrouza/exvivomouse/brainatlas', 'DSURQE_40micron_average_masked.nii')
    flt.inputs.in_file = moving_file_dir
    flt.inputs.reference = static_file_dir
    flt.inputs.output_type = "NIFTI_GZ"
    flt.inputs.searchr_x = [-180, 180]
    flt.inputs.searchr_y = [-180, 180]
    flt.inputs.searchr_z = [-180, 180]
    flt.inputs.dof = 12
    flt.inputs.interp = 'trilinear'
    flt.inputs.out_file = os.path.join(input_data_path, 'masked_first_vtr_linreg.nii.gz')
    flt.inputs.out_matrix_file = os.path.join(input_data_path, 'masked_first_vtr_linreg.mat')
    print(flt.cmdline)
    flt.run()

    # INVERT THE OBTAINED TRANSFORM
    inv_xform = fsl.ConvertXFM()
    inv_xform.inputs.out_file = os.path.join(input_data_path, 'masked_first_vtr_linreg_inv.mat')
    inv_xform.inputs.in_file = os.path.join(input_data_path, 'masked_first_vtr_linreg.mat')
    inv_xform.inputs.invert_xfm = bool(1)
    print(inv_xform.cmdline)
    inv_xform.run()

    # APPLY THE INVERSE TRANSFORM TO THE ATLAS AND LABELS
    app_xform = fsl.ApplyXFM()
    app_xform.inputs.in_file = '/data/rudko/vgrouza/exvivomouse/brainatlas/DSURQE_40micron_labels.nii'
    app_xform.inputs.reference = os.path.join(input_data_path, 'masked_first_vtr.nii.gz')
    app_xform.inputs.padding_size = 0
    #app_xform.inputs.interp = 'trilinear'
    app_xform.inputs.interp = 'nearestneighbour'
    app_xform.inputs.in_matrix_file = os.path.join(input_data_path, 'masked_first_vtr_linreg_inv.mat')
    app_xform.inputs.out_file = os.path.join(input_data_path, 'labels_inv.nii.gz')
    app_xform.inputs.out_matrix_file = 'temp'
    print(app_xform.cmdline)
    app_xform.run()

    # PROVIDE TIMING
    print("Performing inverse registration of template labels...")
    print("Elapsed time is: ", -t + time(), "seconds.")
    return 0


def main(nii_path):
    # LOAD DATA
    data_path = os.path.dirname(nii_path)
    print(data_path)
    data, affine, img = load_nifti(nii_path, return_img=True)

    # PREPROCESS DATA
    corrected_data = preprocessing(data, affine)
    save_nifti(os.path.join(data_path, 'corrected_data.nii.gz'), corrected_data, affine)

    # SEGMENT DATA
    skull_stripping(data_path)

    # DO BIAS FIELD CORRECTION
    bias_field_correcting(data_path)

    # FIT T1 MAP USING LSQNONNEG
    t1map_fitting(data_path)

    # DO AFFINE REGISTRATION TO TEMPLATE WITH FSL-FLIRT
    inverse_reg(data_path)
    return 0


# make a directory list of all rarevtr niftis
os.chdir('/data/rudko/vgrouza/invivomouse/*')
import os
import fnmatch

inDIR = os.getcwd()
pattern = '*.nii.gz'
fileList = []

# Walk through directory
for dName, sdName, fList in os.walk(inDIR):
    for fileName in fList:
        if fnmatch.fnmatch(fileName, pattern):  # Match search string
            fileList.append(os.path.join(dName, fileName))

rarevtr_list = []
for i in range(0, len(fileList)):
    if fileList[i].__contains__("RAREVTR"):
        rarevtr_list.append(fileList[i])
        if fileList[i].__contains__("subscan_1"):
            rarevtr_list.pop()
        if os.path.basename(fileList[i]).__contains__("RAREVTR") == False:
            rarevtr_list.pop()

rarevtr_list = sorted(rarevtr_list)

for i in range(0, len(rarevtr_list)):
    main(rarevtr_list[i])

cohort_0_list = ['/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/170/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/170/niftis/RAREVTR_20/RAREVTR_20.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/174/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/174/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/175/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/175/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/181/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/181/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/354/niftis/RAREVTR_15/RAREVTR_15.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/354/niftis/RAREVTR_20/RAREVTR_20.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/356/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/356/niftis/RAREVTR_20/RAREVTR_20.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/388/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/388/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/432/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/432/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/434/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/434/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/435/niftis/RAREVTR_13/RAREVTR_13.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/435/niftis/RAREVTR_20/RAREVTR_20.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/436/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/436/niftis/RAREVTR_17/RAREVTR_17.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/441/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/441/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/442/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/442/niftis/RAREVTR_17/RAREVTR_17.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/444/niftis/RAREVTR_13/RAREVTR_13.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/444/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/446/niftis/RAREVTR_13/RAREVTR_13.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/446/niftis/RAREVTR_17/RAREVTR_17.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/449/niftis/RAREVTR_15/RAREVTR_15.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/449/niftis/RAREVTR_18/RAREVTR_18.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/476/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/476/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/477/niftis/RAREVTR_15/RAREVTR_15.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_0/477/niftis/RAREVTR_19/RAREVTR_19.nii.gz']
cohort_1_list = ['/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/131/niftis/RAREVTR_19/RAREVTR_19_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/131/niftis/RAREVTR_25/RAREVTR_25_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/155/niftis/RAREVTR_10/RAREVTR_10.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/155/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/261/niftis/RAREVTR_14/RAREVTR_14_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/261/niftis/RAREVTR_18/RAREVTR_18_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/263/niftis/RAREVTR_16/RAREVTR_16_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/263/niftis/RAREVTR_22/RAREVTR_22_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/270/niftis/RAREVTR_25/RAREVTR_25_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/270/niftis/RAREVTR_31/RAREVTR_31_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/271/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/271/niftis/RAREVTR_21/RAREVTR_21_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/274/niftis/RAREVTR_12/RAREVTR_12_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/274/niftis/RAREVTR_17/RAREVTR_17_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/275/niftis/RAREVTR_13/RAREVTR_13.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/275/niftis/RAREVTR_18/RAREVTR_18_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/276/niftis/RAREVTR_10/RAREVTR_10.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/276/niftis/RAREVTR_15/RAREVTR_15.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/428/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/428/niftis/RAREVTR_18/RAREVTR_18_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/433/niftis/RAREVTR_13/RAREVTR_13.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/433/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/435/niftis/RAREVTR_14/RAREVTR_14_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/435/niftis/RAREVTR_20/RAREVTR_20.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/438/niftis/RAREVTR_13/RAREVTR_13_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/438/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/440/niftis/RAREVTR_14/RAREVTR_14_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/440/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/444/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/444/niftis/RAREVTR_18/RAREVTR_18.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/445/niftis/RAREVTR_14/RAREVTR_14_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/445/niftis/RAREVTR_22/RAREVTR_22_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/449/niftis/RAREVTR_15/RAREVTR_15_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/449/niftis/RAREVTR_23/RAREVTR_23_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/453/niftis/RAREVTR_10/RAREVTR_10.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/453/niftis/RAREVTR_15/RAREVTR_15.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/464/niftis/RAREVTR_10/RAREVTR_10.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/464/niftis/RAREVTR_17/RAREVTR_17_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/508/niftis/RAREVTR_17/RAREVTR_17.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/508/niftis/RAREVTR_26/RAREVTR_26.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/509/niftis/RAREVTR_10/RAREVTR_10.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/509/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/513/niftis/RAREVTR_15/RAREVTR_15.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/513/niftis/RAREVTR_21/RAREVTR_21.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/514/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/514/niftis/RAREVTR_23/RAREVTR_23.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/519/niftis/RAREVTR_10/RAREVTR_10.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/519/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/535/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/535/niftis/RAREVTR_22/RAREVTR_22_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/536/niftis/RAREVTR_13/RAREVTR_13.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/536/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/538/niftis/RAREVTR_13/RAREVTR_13_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/538/niftis/RAREVTR_22/RAREVTR_22_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/542/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/542/niftis/RAREVTR_22/RAREVTR_22_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/543/niftis/RAREVTR_12/RAREVTR_12_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/543/niftis/RAREVTR_22/RAREVTR_22_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/544/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/544/niftis/RAREVTR_21/RAREVTR_21.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/592/niftis/RAREVTR_13/RAREVTR_13.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/592/niftis/RAREVTR_25/RAREVTR_25.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/593/niftis/RAREVTR_10/RAREVTR_10.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/593/niftis/RAREVTR_16/RAREVTR_16_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/594/niftis/RAREVTR_13/RAREVTR_13_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/594/niftis/RAREVTR_19/RAREVTR_19_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/664/niftis/RAREVTR_13/RAREVTR_13_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/664/niftis/RAREVTR_20/RAREVTR_20_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/665/niftis/RAREVTR_17/RAREVTR_17.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/665/niftis/RAREVTR_23/RAREVTR_23_subscan_0.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/674/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/674/niftis/RAREVTR_18/RAREVTR_18.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/676/niftis/RAREVTR_13/RAREVTR_13.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/676/niftis/RAREVTR_18/RAREVTR_18.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/677/niftis/RAREVTR_10/RAREVTR_10.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/677/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/678/niftis/RAREVTR_10/RAREVTR_10.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/678/niftis/RAREVTR_21/RAREVTR_21.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/681/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_1/681/niftis/RAREVTR_19/RAREVTR_19_subscan_0.nii.gz']
cohort_2_list = ['/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/698/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/698/niftis/RAREVTR_17/RAREVTR_17.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/702/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/702/niftis/RAREVTR_22/RAREVTR_22.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/703/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/703/niftis/RAREVTR_18/RAREVTR_18.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/706/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/706/niftis/RAREVTR_18/RAREVTR_18.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/707/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/707/niftis/RAREVTR_24/RAREVTR_24.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/711/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/711/niftis/RAREVTR_21/RAREVTR_21.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/713/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/713/niftis/RAREVTR_18/RAREVTR_18.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/725/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/725/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/726/niftis/RAREVTR_13/RAREVTR_13.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/726/niftis/RAREVTR_17/RAREVTR_17.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/727/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/727/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/729/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/729/niftis/RAREVTR_22/RAREVTR_22.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/731/niftis/RAREVTR_15/RAREVTR_15.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/731/niftis/RAREVTR_20/RAREVTR_20.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/733/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/733/niftis/RAREVTR_28/RAREVTR_28.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/734/niftis/RAREVTR_15/RAREVTR_15.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/734/niftis/RAREVTR_24/RAREVTR_24.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/735/niftis/RAREVTR_18/RAREVTR_18.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/735/niftis/RAREVTR_23/RAREVTR_23.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/777/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/777/niftis/RAREVTR_18/RAREVTR_18.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/778/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/778/niftis/RAREVTR_23/RAREVTR_23.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/779/niftis/RAREVTR_18/RAREVTR_18.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/779/niftis/RAREVTR_24/RAREVTR_24.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/782/niftis/RAREVTR_29/RAREVTR_29.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/782/niftis/RAREVTR_39/RAREVTR_39.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/784/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/784/niftis/RAREVTR_30/RAREVTR_30.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/788/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/788/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/812/niftis/RAREVTR_13/RAREVTR_13.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/812/niftis/RAREVTR_20/RAREVTR_20.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/817/niftis/RAREVTR_13/RAREVTR_13.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/817/niftis/RAREVTR_24/RAREVTR_24.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/818/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/818/niftis/RAREVTR_21/RAREVTR_21.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/821/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/821/niftis/RAREVTR_24/RAREVTR_24.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/825/niftis/RAREVTR_15/RAREVTR_15.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/825/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/833/niftis/RAREVTR_14/RAREVTR_14.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/833/niftis/RAREVTR_18/RAREVTR_18.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/834/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/834/niftis/RAREVTR_19/RAREVTR_19.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/838/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/838/niftis/RAREVTR_16/RAREVTR_16.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/839/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/839/niftis/RAREVTR_22/RAREVTR_22.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/843/niftis/RAREVTR_15/RAREVTR_15.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/843/niftis/RAREVTR_24/RAREVTR_24.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/845/niftis/RAREVTR_12/RAREVTR_12.nii.gz', '/data/rudko/vgrouza/invivomouse/pddata/data/cohort_2/845/niftis/RAREVTR_21/RAREVTR_21.nii.gz']

