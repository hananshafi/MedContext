import torch
from torch import nn
from monai.data.utils import dense_patch_slices
import numpy as np

def quantize_data(data):
	device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
	rand_num = torch.randint(1,4,data.shape).to(device)
	return torch.floor(data / rand_num)

def block_splitting(image, block_size = (8,8,8)):

    """ Splitting 3D volume into smaller cubes of block_size

    Input:

        image: [B,C,H,W,D]



    Output:

        image_new (sub_blocks):  [B,C, N_Blocks, block_size[0], block_size[1], block_size[2] ]   where N_Blocks=(H*W*D)//np.prod(block_size)

    """



    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")


    B,C,H,W,D = image.shape



    image_size = (H,W,D)

    roi_size = block_size

    scan_interval = block_size

    slices = dense_patch_slices(image_size, roi_size, scan_interval)



    num_blocks = (H*W*D)//(np.prod(block_size))



    assert H%block_size[0]==0 , f"Height of image should be divisible by block_size[0]"

    assert W%block_size[1]==0 , f"Width of image should be divisible by block_size[1]"

    assert D%block_size[2]==0 , f"Depth of image should be divisible by block_size[2]"



    assert len(slices)==num_blocks , f"Number of 3D sub-blocks={len(slices)} of given image is not equal to expected number of sub-bloks={num_blocks}"



    image_new_shape = (B,C,num_blocks)+block_size

    image_new = torch.zeros(image_new_shape, dtype=torch.float32).to(device)



    for b_dim in range(B): # batch dimension

        for c_dim in range(C): # channel dimension

            for i, i_slice in enumerate(slices):

                image_new[b_dim,c_dim,i] = image[b_dim,c_dim][i_slice]



    return image_new


def block_merging(image, new_image_shape):

    """ Merge smaller cubes of block_size into big 3D volume



    Input:

        image (sub_blocks): [B,C, N_Blocks, block_size[0], block_size[1], block_size[2] ]   where N_Blocks=(H*W*D)//np.prod(block_size)

        new_image_shape   : desired shape of the new image [B,C,H,W,D]

    Output:

        image_new:  [B,C,H,W,D]



    """



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    B,C,num_blocks,block_h,block_w,block_d = image.shape



    block_size = (block_h,block_w,block_d)



    H=new_image_shape[2]

    W=new_image_shape[3]

    D=new_image_shape[4]



    



    image_size = (H,W,D)

    roi_size = block_size

    scan_interval = block_size

    slices = dense_patch_slices( image_size, roi_size, scan_interval)



    assert len(slices)==num_blocks , f"Cannot merge sub-blocks of given image into new image of desired shape due to conflict in sizes."



     

    image_new = torch.zeros(new_image_shape, dtype=torch.float32).to(device) # [B,C,H,W,D]



    for b_dim in range(B): # batch dimension

        for c_dim in range(C): # channel dimension

            for i, i_slice in enumerate(slices):

                image_new[b_dim,c_dim][i_slice] = image[b_dim,c_dim][i]

                

    return image_new
