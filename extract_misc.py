import numpy as np
import nibabel as nib
import random, re, glob
import SimpleITK as sitk
import pdb, os, time
from tqdm import tqdm, trange
import pandas as pd
from scipy.ndimage import zoom

# patch_size = [224, 224] #  [256, 256]

def data_zoom(data, img_reso, resolution=(1.0, 1.0, 1.0)):
  
  from cupyx.scipy.ndimage import zoom
  import cupy as cp

  zoom_factors = (img_reso[0] / resolution[0], img_reso[1] / resolution[1], img_reso[2] / resolution[2])
  if data.dtype in [np.float32, np.float64]:
    data_zoomed = zoom(cp.array(data, dtype=cp.float64), zoom_factors, order=3)
    data_zoomed = cp.asnumpy(data_zoomed).astype(np.float32)
  else:
    data_zoomed = zoom(cp.array(data, dtype=cp.int64), zoom_factors, order=0)
    data_zoomed = cp.asnumpy(data_zoomed).astype(np.uint8)

  return data_zoomed


def adjust_image_size(img_array, patch_size, axes):
    new_size = list(img_array.shape)
    for i, axis in enumerate(axes):
        if new_size[axis] < patch_size[i]:
            new_size[axis] = patch_size[i] + 1
    new_img_array = np.flip(img_array, axis=axes) if any(s > os for s, os in zip(new_size, img_array.shape)) else img_array
    new_img_array = np.pad(new_img_array, [(0, ns - s) for s, ns in zip(img_array.shape, new_size)])
    return new_img_array


def adjust_slices(slices, array_shape, patch_size):
    # スライスの開始と終了のインデックスを抽出
    start_y, stop_y = slices[0].start, slices[0].stop
    start_x, stop_x = slices[1].start, slices[1].stop
    
    # スライスがarrayの境界を超えないように調整
    if stop_y > array_shape[0]:
        start_y -= (stop_y - array_shape[0])
        stop_y = array_shape[0]
    if stop_x > array_shape[1]:
        start_x -= (stop_x - array_shape[1])
        stop_x = array_shape[1]
    
    # もしスタートが負の値になった場合はゼロに修正
    start_y = max(0, start_y)
    start_x = max(0, start_x)
    
    # スライスのサイズがpatch_sizeと異なる場合は調整
    if (stop_y - start_y) != patch_size[0]:
        stop_y = start_y + patch_size[0]
    if (stop_x - start_x) != patch_size[1]:
        stop_x = start_x + patch_size[1]
    
    # スライスがarrayの境界を超えていないことを再確認
    stop_y = min(stop_y, array_shape[0])
    stop_x = min(stop_x, array_shape[1])
    
    # 調整されたスライスを返す
    return (slice(start_y, stop_y), slice(start_x, stop_x), slices[2])


def extract_patches2(img_array, label_array, patch_size, n_patches=1000, axes=(0,1), pos_ratio=0.9, prev_overlap=0.2, sim_dict=None, img_name="Unknown", debug=False, overlap_array=None):
    
    patch_info = pd.DataFrame(columns=['patch_idx', 'patch_position', 'label', 'overlap_ratio', 'loop_counter', 'start_x', 'end_x', 'start_y', 'end_y', 'start_z', 'end_z'])

    # Ensure correct image sizes
    img_array = adjust_image_size(img_array, patch_size, axes)
    label_array = adjust_image_size(label_array, patch_size, axes)

    # Preallocate the memory for patches
    img_patches = np.empty((n_patches, *patch_size), dtype=np.float32)
    label_patches = np.empty((n_patches, *patch_size), dtype=np.uint8)

    if sim_dict is not None:
       sim_dict['array']   = adjust_image_size(sim_dict['array'], patch_size, axes)
       sim_dict['patches'] = np.empty((n_patches, *patch_size), dtype=np.uint8)

    # Create cropped region
    cropped_region = np.zeros_like(label_array)

    count_patches = 0
    n_pos = 0
    n_neg = 0
    finish_flag = False

    for patch_idx in tqdm(range(n_patches), desc=img_name, leave=False): # extra iteration to avoid infinite loop
        
        # Determine whether to find a patch with a label or not
        with_label = np.random.rand() < pos_ratio
        
        # Find a patch position with a label
        loop_counter = 0
        while True:
            # Determine non-axes direction
            non_axes = [ax for ax in range(3) if ax not in axes]

            # Get random position in non-axes direction
            non_axes_pos = [np.random.randint(0, label_array.shape[ax]) for ax in non_axes]

            # Use overlap_array to determine the patch center
            if overlap_array is not None:
                # Flatten the overlap_array and normalize it to create a probability distribution
                # prob_dist = overlap_array.flatten() / overlap_array.max()

                # Randomly select an index based on the probability distribution
                center_idx = np.random.choice(np.arange(overlap_array.size)) # , p=prob_dist)

                # Convert the flattened index back to 3D coordinates
                center_coords = np.unravel_index(center_idx, overlap_array.shape)

                # Calculate the patch start and end positions based on the center coordinates
                patch_position = [center_coords[ax] - patch_size[ax] // 2 if ax in axes else non_axes_pos[0] for ax in range(3)]
            else:
                # Get random position (original method)
                patch_position = [np.random.randint(0, s - patch_size[ax] + 1) if ax in axes else non_axes_pos[0] for ax, s in enumerate(label_array.shape)]

            # Create slices object for patch extraction
            slices = [slice(p, p + patch_size[ax]) if ax in axes else slice(p, p + 1) for ax, p in enumerate(patch_position)]
            slices = tuple(slices)
            cropped = label_array[slices] if sim_dict is None else sim_dict["array"][slices]


            # Check if this patch has a label or not
            if with_label and np.any(cropped > 0):
                # Check if the patch overlaps significantly with the already cropped region
                overlap_ratio = np.mean(cropped_region[slices] > 0)
                if overlap_ratio <= prev_overlap:
                  n_pos += 1
                  break
            elif not with_label and np.all(cropped == 0):
                # Check if the patch overlaps significantly with the already cropped region
                overlap_ratio = np.mean(cropped_region[slices] > 0)
                if overlap_ratio <= prev_overlap:
                  n_neg += 1
                  break
            
            loop_counter += 1

            if loop_counter > 950:
                print(non_axes, non_axes_pos, patch_position)
                print("Max attempts reached for finding suitable patch position. Finished with", count_patches, "patches.")
                finish_flag = True
                break # return None, None

                #if loop_counter > 960:
                #  print("Max attempts reached for finding suitable patch position. You may want to adjust parameters.")
                  # import pdb; pdb.set_trace()
            #if loop_counter > 1000: # Add maximum number of attempts to avoid infinite loop
            #    overlap_ratio = 0
            #    break # return None, None

        # Extract patch from image and label array
        slices = adjust_slices(slices, img_array.shape, patch_size=patch_size)
        try:
          img_patches[count_patches] = np.squeeze(img_array[slices])
        except Exception as e:
          import traceback; traceback.print_exc()
          import pdb; pdb.set_trace()


        # label_patches[count_patches] = np.squeeze(label_array[slices]) if sim_dict is None else np.squeeze(sim_dict["array"][slices])
        label_patches[count_patches] = np.squeeze(label_array[slices])
        if sim_dict is not None:
            sim_dict["patches"][count_patches] = np.squeeze(sim_dict["array"][slices])[np.newaxis, ...]
        
        try:
          patch_info.loc[patch_idx] = [count_patches, patch_position, with_label, overlap_ratio, loop_counter, slices[0].start, slices[0].stop, slices[1].start, slices[1].stop, slices[2].start, slices[2].stop]
        except Exception as e:
           import traceback; traceback.print_exc()
           pdb.set_trace()

        # Update the cropped region
        cropped_region[slices] = 1
        count_patches += 1

        # If the desired number of patches has been extracted, break the loop
        if count_patches >= n_patches or finish_flag:
            break

    print("n_pos:", n_pos, "n_neg:", n_neg)
    # If less patches were extracted, adjust the size of the patch arrays
    if count_patches < n_patches:
        img_patches = img_patches[:count_patches]
        label_patches = label_patches[:count_patches]
        if sim_dict is not None:
            sim_dict["patches"] = sim_dict["patches"][:count_patches]
    
    return img_patches, label_patches, patch_info, sim_dict





def extract_patches(img_array, label_array, patch_size):

    patches_img = []
    patches_label = []

    # Initialize lists to store patches

    for sliceIdx in range(img_array.shape[2]):
        oneSliceImage = img_array[..., sliceIdx]
        oneSliceLabel = label_array[..., sliceIdx]

        # If any dimension is smaller than the patch size, pad it to the required size
        for i in range(2):
            if oneSliceImage.shape[i] < patch_size[i]:
                pad_width = max(patch_size[i] - oneSliceImage.shape[i], patch_size[i] // 2)
                oneSliceImage = np.pad(oneSliceImage, ((pad_width, pad_width) if i == 0 else (0, 0),
                                                        (pad_width, pad_width) if i == 1 else (0, 0)), mode='reflect')
                oneSliceLabel = np.pad(oneSliceLabel, ((pad_width, pad_width) if i == 0 else (0, 0),
                                                       (pad_width, pad_width) if i == 1 else (0, 0)), mode='reflect')
        
        positivePositions = np.array(np.where(oneSliceLabel>0)).T
        if positivePositions.shape[0]==0:
           continue

        center = np.mean(positivePositions, axis=0).astype(np.int32)
        # Make sure that all patches does not cut outside the original slice
        for dimIdx in range(len(patch_size)):
          if center[dimIdx] < int(patch_size[dimIdx]/2):
            center[dimIdx] = int(patch_size[dimIdx]/2)
          elif center[dimIdx] > oneSliceImage.shape[dimIdx] - int(patch_size[dimIdx]/2) - 1 :
            center[dimIdx] = oneSliceImage.shape[dimIdx] - int(patch_size[dimIdx]/2) - 1

        patch_img   = oneSliceImage[center[0]-int(patch_size[0]/2):center[0]+int(patch_size[0]/2), center[1]-int(patch_size[1]/2):center[1]+int(patch_size[1]/2)] 
        patch_label = oneSliceLabel[center[0]-int(patch_size[0]/2):center[0]+int(patch_size[0]/2), center[1]-int(patch_size[1]/2):center[1]+int(patch_size[1]/2)]

        if tuple(patch_img.shape) != tuple(patch_size): # error
           print(patch_img.shape)
           pdb.set_trace()

        # Append patches to lists
        patches_img.append(patch_img)
        patches_label.append(patch_label)

    return patches_img, patches_label


def cutoff_top_percentile(image, percentile=98):
    """Cut off the top 2% of the intensity histogram."""
    cutoff_threshold = np.percentile(image, percentile)
    image_cropped = np.clip(image, None, cutoff_threshold)
    
    return image_cropped

   
def normalize_image(image, modal):
    """Normalize the intensity values of the image"""

    image = cutoff_top_percentile(image)

    target_range = (-200, 500) if modal == "ct" else (0, 3000)
    min_val, max_val = target_range

    normalized_image = (image.astype(np.float32) if image.dtype != np.float32 else image)
    normalized_image = np.clip(normalized_image, min_val, max_val)
    normalized_image = (normalized_image - min_val) / (max_val - min_val)
    
    return normalized_image


def rescale_image(img_data, original_voxel_size, resolution):
    """Rescale the image to the new voxel size based on the desired resolution"""
    new_voxel_size = tuple(res / resolution for res in original_voxel_size)
    return zoom(img_data, new_voxel_size)


def divide_image(img_data, n_div=3, slice_z=1):
    """Divide the image data into slices"""
    # Find the dimensions of the image data
    x_len, y_len, z_len = img_data.shape

    # Define the x and y coordinates for slicing
    x_coords = np.linspace(0, x_len-1, n_div, dtype=int)
    y_coords = np.linspace(0, y_len-1, n_div, dtype=int)

    slices = []
    for z in range(0, z_len, slice_z):
        for x in x_coords:
            for y in y_coords:
                x_begin = x
                x_end = x_begin + 256 if x_begin + 256 < x_len else x_len
                y_begin = y
                y_end = y_begin + 256 if y_begin + 256 < y_len else y_len
                slice_3d = img_data[x_begin:x_end, y_begin:y_end, z:z+slice_z]
                slices.append(((x_begin, x_end), (y_begin, y_end), (z, z+slice_z), slice_3d))
    return slices

import numpy as np

def pad_array(img_data, target_size):
    # Get the original shape of img_data
    original_shape = img_data.shape

    # Calculate the deficit to reach the target size
    deficit = np.maximum(np.subtract(target_size, original_shape[:2]), 0)

    # Check if padding is needed
    if np.any(deficit > 0):
        # Generate an array of flipped and deficit-sized padding
        padding = np.flip(img_data[:deficit[0], :deficit[1]], axis=(0, 1))
        
        # Pad the array along the first and second dimensions (0th and 1st axes)
        padded_img_data = np.pad(img_data, ((0, deficit[0]), (0, deficit[1]), (0, 0)), mode='constant')
        
        # Replace the deficit-sized region with the flipped padding
        padded_img_data[:deficit[0], :deficit[1]] = padding

        print("Padding img data", img_data.shape, "-->", padded_img_data.shape)
    else:
        # No padding needed, return the original img_data
        padded_img_data = img_data

    return padded_img_data

def slice_nifti(img_data, patch_size = (256, 256, 1), n_div=3):
    """Slice the 3D image data into smaller volumes"""

    padded_img_data = pad_array(img_data, patch_size[0:2])
    max_x, max_y, max_z = padded_img_data.shape

    x_coords = np.linspace(0, max_x - patch_size[0], n_div, dtype=int)
    y_coords = np.linspace(0, max_y - patch_size[1], n_div, dtype=int)
    # z_coords = np.linspace(0, max_z - patch_size[2], n_div, dtype=int)
    z_coords = range(0, max_z - patch_size[2])

    num_slices = len(x_coords) * len(y_coords) * len(z_coords)
    
    slices = [None] * num_slices

    idx = 0


    for x_begin in tqdm(x_coords, leave=False):
        x_end = x_begin + patch_size[0]

        for y_begin in y_coords:
            y_end = y_begin + patch_size[1]

            for z_begin in z_coords:
                z_end = z_begin + patch_size[2]

                slice_3d = padded_img_data[x_begin:x_end, y_begin:y_end, z_begin:z_end]

                try:
                  slice_3d = slice_3d.reshape(patch_size).astype(img_data.dtype)
                except Exception as exc:
                  print(slice_3d.shape)
                  pdb.set_trace()

                slices[idx] = ((x_begin, x_end), (y_begin, y_end), (z_begin, z_end), slice_3d)
                idx += 1

    return slices


def save_slices(slice_info, base_name, affine, sequence_number, output_subdir_path, idx, mode="train", modal="ct"):
    (x_begin, x_end), (y_begin, y_end), (z_begin, z_end), slice_3d = slice_info  # Unpack the tuple to get the 3D slice 

    # Form the output file name
    prefix = "" if mode=="train" else "val"
    output_file = f"{prefix}{modal}slice{sequence_number}_{idx:04d}.nii"

    # Create the Nifti1Image
    slice_img = nib.Nifti1Image(slice_3d.transpose((2,0,1)), affine)

    # Save the slice to a Nifti file
    nib.save(slice_img, os.path.join(output_subdir_path,  output_file))
    return output_file


def save_slices_to_csv(slices, output_file_paths, base_name, output_subdir_path, sequence_number):
    """Save the information about the slices to a CSV file"""
    csv_path = os.path.join(output_subdir_path, base_name.replace("_label.nii.gz", "") + '.csv')
    print(csv_path)

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ["input_file", "patch_file", "x_begin", "x_end", "y_begin", "y_end", "z_begin", "z_end", "label_sizes"]
        import csv
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for idx, ((x_begin, x_end), (y_begin, y_end), (z_begin, z_end), _) in enumerate(slices):
            label_sizes = [np.sum(slices[idx][3]==lbl) for lbl in range(1, 8)]
            # patch_file = (f"{base_name}_{idx}.nii").replace(".nii.gz","")
            patch_file = output_file_paths[idx] # (f"{base_name}_{idx}.nii").replace(".nii.gz","")
            writer.writerow({
                "input_file": base_name,
                "patch_file": patch_file,
                "x_begin": x_begin,
                "x_end": x_end,
                "y_begin": y_begin,
                "y_end": y_end,
                "z_begin": z_begin,
                "z_end": z_end,
                "label_sizes": label_sizes
            })


def divide_nifti(input_dir_files, output_dir, resolution=(0.5, 0.5, 0.5), patch_size=(256,256,1), overlap=1/4, mode="train", modal="ct"):

    """Divides 3D Nifti images into smaller 3D patches and save the information about slices to a CSV file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List the files in the input directory
    # input_dir_files = glob.glob(os.path.join(input_dir, '*.nii.gz'))
    
    # Determine the output subdirectories
    output_subdirs = ["IMG", "GT"]
    for subdir in output_subdirs:
        subdir_path = os.path.join(output_dir, ("train" if mode=="train" else "val"), subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)

    for file_path in tqdm(input_dir_files, desc="Dividing images", leave=False ):
        base_name = os.path.basename(file_path)
        subdir = "IMG" if "_image.nii.gz" in base_name else "GT"
        sequence_number = int(re.search(r'\d+', base_name).group())
        
          
        img = nib.load(file_path)
        img = nib.as_closest_canonical(img)  # Convert to RAS orientation
        data = img.get_fdata().astype(np.float32 if "_image.nii.gz" in base_name else np.uint16)
        if data.dtype==np.float32:
          data = normalize_image(data, modal)
        elif np.max(data)>170:
          data = value_mapping(data).astype(np.uint8)
          print(np.unique(data))
        else:
          data = data.astype(np.uint8)

        # Perform cubic interpolation to adjust resolution
        img_reso = img.header.get_zooms() # np.diag(np.abs(img.affine))
        if img_reso != resolution:
          print("Resampling to 0.5mm resolution from", img_reso)
          data_zoomed = data_zoom(data, img_reso)

        slices = slice_nifti(data_zoomed, patch_size)

        output_subdir_path = os.path.join(output_dir, ("train" if mode=="train" else "val"), subdir)
        
        print()
        print("Started writing slices")
        output_file_paths = [save_slices(slice_info, base_name, out_affine, sequence_number, output_subdir_path, idx, mode, modal) \
                             for idx, slice_info in tqdm(enumerate(slices), total=len(slices), desc=f"Saving patches from {base_name}", leave=False) ]

        save_slices_to_csv(slices, output_file_paths, base_name, output_subdir_path, sequence_number)

    print("Finished dividing all Nifti images.")

def save_as_nifti(array, affine, save_path):
    # Create a new nifti object
    new_nifti = nib.Nifti1Image(array, affine) # , original_nifti.header)

    # Save the nifti object
    nib.save(new_nifti, save_path)

def weak_tagging(patch_label, mean_sizes):
    result = []
    unique_labels = np.unique(patch_label)
    for label in range(0, len(patch_label) + 1 ):
        if label in unique_labels:
            result.append(mean_sizes[label - 1])
        else:
            result.append(0)
    return result

def value_mapping(input, values=[500,600,420,550,205,820,850]):
    output = np.copy(input)  # Create a copy of the input
    for value in values:
        output[input == value] = values.index(value) + 1
    return output

def calc_mean_sizes(arr_list, min_val=0, max_val=7):
    
    if isinstance(arr_list, np.ndarray):
        arr_list = [arr_list]

    count_list = []
    for i in range(min_val, max_val + 1):
        count = 0
        for arr in arr_list:
            count += np.count_nonzero(arr == i)
        if len(arr_list)==1:
           count_list.append(count)
        else:
          count_list.append(count / len(arr_list))

    return count_list


def calc_average_list(df, col_name = 'val_gt_size'):

  def calculate_average_at_position(lst, position):
      values_at_position = [x[position] for x in lst]
      return sum(values_at_position) / len(values_at_position)
  try:
    # Calculate the average value of each element at each position in col_name column and store the results in a list
    num_elements = len(df[col_name][0])
    average_list = [calculate_average_at_position(df[col_name], i) for i in range(num_elements)]
  except Exception as e:
    print("Error in calc_average_list", e)
    pdb.set_trace()

  return average_list

def round_floats_in_list(lst):
    return [round(float_item, 2) for float_item in lst]



def rotate_axis(img_array, plane_orientation):
    """
    Rotate the image data to the specified plane orientation.
    
    Parameters:
    - img_array: numpy.ndarray, the input image data.
    - plane_orientation: str, the target plane orientation ('axial', 'coronal', 'sagittal').
    
    Returns:
    - rotated_img_array: numpy.ndarray, the rotated image data.
    """
    if plane_orientation == "axial":
        # For axial plane, use the original array as is
        rotated_img_array = img_array
    elif plane_orientation == "coronal":
        rotated_img_array = np.transpose(img_array, (0, 2, 1))
        # rotated_img_array = np.flip(rotated_img_array, 1)  # Flip along the y-axis
    elif plane_orientation == "sagittal":
        rotated_img_array = np.transpose(img_array, (1, 2, 0))
        # rotated_img_array = np.flip(rotated_img_array, 1)  # Flip along the y-axis
    else:
        raise ValueError(f"Unsupported plane_orientation: {plane_orientation}")
    
    return rotated_img_array


def extract(input_dir, img_files, modal, mode, out_base_dir, df_train=None, fold=0, resolution=(0.75, 0.75, 0.75), debug=False, mean_gt_size=None, sim_dir=None, csv_suffix="", perform_save=True, overlap_dir=None, patch_size=(224,224), plane="axial"):

  # Output directory
  # Following some famous methods' style, testing datasets are extracted to "val" directory.
  print(modal, ",", mode, ":", len(img_files), "files are used")
  patch_dir = os.path.join( out_base_dir + "_" + plane, modal, "train" if mode=="train" else "val" )
  os.makedirs(os.path.join(patch_dir, "IMG"), exist_ok=True)
  os.makedirs(os.path.join(patch_dir, "GT"), exist_ok=True)
  if sim_dir is not None:
     os.makedirs(os.path.join(patch_dir, "SIMGT"), exist_ok=True)

  # Saving file list
  with open( os.path.join(out_base_dir+ "_" + plane, "list_"+ modal + "_" + mode + ".txt" ), 'w') as fp:
    _ = [fp.write(os.path.basename(file) + "\n") for file in img_files]

  # data frame for sizes
  if sim_dir is not None:
    df = pd.DataFrame(columns=['input_file', 'val_ids','patch_idx', 'patch_position', 'label', 'start_x', 'end_x', 'start_y', 'end_y', 'start_z', 'end_z', 'overlap_ratio', 'loop_counter', 'val_gt_size', 'val_sim_size', 'dumbpredwtags'])
  else:
    df = pd.DataFrame(columns=['input_file', 'val_ids',  'patch_idx', 'patch_position', 'label', 'start_x', 'end_x', 'start_y', 'end_y', 'start_z', 'end_z', 'overlap_ratio', 'loop_counter', 'val_gt_size', 'dumbpredwtags'])
  patch_infos = []

  serial = 0
  for img_file in tqdm(img_files, desc="[" + modal + "," + mode + "]"):
      
    label_file = img_file.replace('_image.nii.gz', '_label.nii.gz')
    
    # Read nifti files
    img_nifti = nib.load(os.path.join(input_dir, img_file))
    img_nifti = nib.as_closest_canonical(img_nifti)

    label_nifti = nib.load(os.path.join(input_dir, label_file))
    label_nifti = nib.as_closest_canonical(label_nifti)

    img_reso = img_nifti.header.get_zooms()
    
    img_array = np.array(img_nifti.dataobj).astype(np.float32)
    label_array = np.array(label_nifti.dataobj)

    img_array = rotate_axis(img_array, plane)
    label_array = rotate_axis(label_array, plane)

    if np.isin(label_array, [500,600,420,550,205,820,850]).any():
       print("Value mapping performed")
       label_array = value_mapping(label_array).astype(np.uint8)

    if tuple(img_reso) != tuple(resolution):
      print("Resampling to resolution", resolution, "from", img_reso)
      img_array = data_zoom(img_array, img_reso)
      label_array = data_zoom(label_array, img_reso)
    

    if overlap_dir is not None:
       
       if img_file.find("ct_t")>=0:
         overlap_path = os.path.join(overlap_dir.replace("mr","ct"), "0vx", os.path.basename(img_file).replace("image.nii.gz","") + "label_05mm_simulated.nii.gz")
       else:
         overlap_path = os.path.join(overlap_dir.replace("ct","mr"), "0vx", os.path.basename(img_file).replace("image.nii.gz","") + "label_05mm_simulated.nii.gz")
       overlap_nii = nib.load(overlap_path)

       
       overlap_array = np.array(overlap_nii.dataobj).astype(np.float32)
       overlap_array /= 1.0 * len(img_files)
       if overlap_array.shape != img_array.shape:
          overlap_array = data_zoom(overlap_array, img_reso)
       print("overlap_array loaded (actually useless)")
    else:
       overlap_array = None

    if sim_dir is not None:
      
      if overlap_dir is not None:
        sim_paths = [overlap_path]
      else:
        sim_paths = glob.glob(os.path.join(sim_dir, label_file.replace('_label.nii.gz', '_label_05mm_simulated*.nii.gz')))

      if len(sim_paths)==0:
        print("No simulated label found for", label_file)
        import pdb; pdb.set_trace()

      sim_nifti = nib.as_closest_canonical(nib.load(sim_paths[0]))
      sim_array = np.array(sim_nifti.dataobj) if label_array.shape == sim_nifti.dataobj.shape else data_zoom(np.array(sim_nifti.dataobj), img_reso)
    
    sim_dict = {"nifti": sim_nifti, "array": sim_array, "dir": sim_dir} if sim_dir is not None else None

    # mrslice_1010 contains value 421, represented as 165 with np.uint8 here 
    # if np.sum(np.isin(np.unique(label_array), [0,1,2,3,4,5,6,7])==False)>0:
    if np.max(label_array) > 10:
      print("Unknown value exists:", np.unique(label_array), "in", os.path.basename(label_file))
      if debug:
         import pdb; pdb.set_trace()
      else:
         label_array = np.where((label_array >= 0) & (label_array <= 7), label_array, 0)
         print("Updated:", np.unique(label_array))
    
    # Normalize image
    img_array = normalize_image(img_array, modal)

    # Extract patches
    patches_img, patches_label, patch_info, sim_dict = extract_patches2(img_array, label_array, patch_size=patch_size, sim_dict=sim_dict, img_name=img_file.replace(".nii.gz",""), n_patches=(30 if debug else 1000), debug=debug )#  , overlap_array=overlap_array )

    # Save patches
    for i, (patch_img, patch_label) in enumerate(tqdm(zip(patches_img, patches_label), desc="Saving patches:" + img_file, leave=False)):

        # Save patches
        out_nifti = np.diag((resolution[0], resolution[1], resolution[2], 1.0))
        out_name = ("val" if mode=="test" else "" ) + modal + "slice" + str(serial) + "_1.nii"
        patch_img_out = np.repeat(patch_img.reshape((1, patch_size[0], patch_size[1], 1)), 3, axis=3)

        if perform_save:
          save_as_nifti(patch_img[np.newaxis, ...].astype(np.float32), out_nifti, os.path.join(patch_dir, "IMG", out_name ))
          save_as_nifti(patch_label[np.newaxis, ...].astype(np.uint8), out_nifti, os.path.join(patch_dir, "GT", out_name))

        if sim_dict is not None: # this should be performed regardless of perform_save
            save_as_nifti(sim_dict["patches"][i, np.newaxis, :, :].astype(np.uint8), out_nifti, os.path.join(patch_dir, "SIMGT", out_name))

        val_gt_size = calc_mean_sizes(patch_label)
        new_data = {'input_file': os.path.basename(img_file).replace(".nii.gz", ""),
                    'val_ids': out_name,
                    'val_gt_size': val_gt_size,
                    'dumbpredwtags': (val_gt_size if mean_gt_size is None else mean_gt_size)
        }
                    # 'dumbprednotags': [np.sum(val_gt_size) // len(val_gt_size) for _ in range(len(val_gt_size))]}  # dummy values
        
        if sim_dict is not None:
          val_sim_size = calc_mean_sizes(sim_dict["patches"][i, ...])
          new_data['val_sim_size'] = val_sim_size
        new_data.update(patch_info.iloc[i].to_dict())

        # df = df.append(new_data, ignore_index=True)
        try:
          df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
        except Exception as exc:
          print("Error in pd.concat", exc)
          import pdb; pdb.set_trace()
        serial += 1

  # Sizes for "dumbpredwtags" and "dumbprednotags"
  if mode=="train":
    print("Size calculation from this dataset")
    df_here = df 
  else:
    print("Size calculation from training dataset")
    df_here = df_train

  df["dumbprednotags"] = [calc_average_list(df_here)]*len(df)

  df['dumbpredwtags'] = [[x[n] if size[n] > 0 else 0 for n in range(len(x))] for x, size in zip(df['dumbprednotags'], df['val_gt_size'])]

  # round floats -> nnnnn.n style, and convert to string
  df['val_gt_size'] = df['val_gt_size']
  df['dumbpredwtags']  = df['dumbpredwtags'].apply(round_floats_in_list)
  df['dumbprednotags'] = df['dumbprednotags'].apply(round_floats_in_list)
  
  # Write size file 
  df.to_csv(os.path.join('sizes', "whs" + str(fold) + "_" + modal + csv_suffix + '.csv'), index=False, sep=";", mode="a" if mode in ("test", "val") else "w", header=(mode=="train"))
  return df_here


def split_img_paths(img_files, n, rate=0.8, seed=0, use_valid=False):
    
    from sklearn.model_selection import KFold

    # Split img_files into training and test data based on n splits and the given rate
    kfold = KFold(n_splits=n, shuffle=True, random_state=seed)
    
    split_data = []
    
    # For each split
    for train_index, test_index in kfold.split(img_files):
        train_data = np.array(img_files)[train_index]
        test_data = np.array(img_files)[test_index]
        
        # If validation data is used, split the training data further into training and validation data
        if use_valid:
            split_point = int(len(train_data) * rate)
            train_data, validation_data = train_data[:split_point], train_data[split_point:]
            split_data.append([train_data.tolist(), test_data.tolist(), validation_data.tolist()])
        else:
            split_data.append([train_data.tolist(), test_data.tolist()])

    # Return the list of training, test, (and possibly validation) data
    return split_data
