import argparse, pathlib
import glob, pdb
import os
import sys
import numpy as np
import torch
import nibabel as nib
import torch.nn as nn
from torch.nn import functional as F
import os.path as osp
from networks import UNet
from tqdm import tqdm, trange
from scipy.ndimage import generate_binary_structure, generic_filter
from collections import Counter
from scipy.ndimage import binary_closing, generate_binary_structure, binary_fill_holes
from termcolor import colored, cprint
import pandas as pd

from extract_misc import rotate_axis, normalize_image


def evaluate_segmentation(output_array, gt_array, filename, n_classes=4):
    """
    Updated evaluation function that also calculates and includes mean metrics for precision, recall, and dice score,
    along with individual class metrics, all rounded to 3 decimal places.

    Args:
    - output_array (np.array): The segmentation result, a numpy array of shape (H, W, D) and type np.uint8.
    - gt_array (np.array): The ground truth segmentation, a numpy array of shape (H, W, D) and type np.uint8.
    - filename (str): The filename for the image being evaluated.
    - n_classes (int): Number of classes including background.

    Returns:
    - pd.DataFrame: A dataframe with the evaluation results, including individual class metrics and mean metrics.
    """
    # Initialize dictionaries to hold TP, FP, FN, and metrics for each class
    metrics = {f"Class{i}-{metric}": 0 for i in range(1, n_classes + 1) for metric in ["TP", "FP", "FN", "Prec", "Recall", "Dice"]}
    precisions = []
    recalls = []
    dices = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Calculate metrics for each class
    for i in range(1, n_classes+1):
        tp = np.sum((output_array == i) & (gt_array == i))
        fp = np.sum((output_array == i) & (gt_array != i))
        fn = np.sum((output_array != i) & (gt_array == i))
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = round(tp / (tp + fp) if tp + fp > 0 else 0, 3)
        recall = round(tp / (tp + fn) if tp + fn > 0 else 0, 3)
        dice = round((2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0, 3)
        
        # Update metrics dictionary
        metrics[f"Class{i}-TP"] = tp
        metrics[f"Class{i}-FP"] = fp
        metrics[f"Class{i}-FN"] = fn
        metrics[f"Class{i}-Prec"] = precision
        metrics[f"Class{i}-Recall"] = recall
        metrics[f"Class{i}-Dice"] = dice
        
        # Append class metrics for calculating means
        precisions.append(precision)
        recalls.append(recall)
        dices.append(dice)

    # Calculate and add mean metrics
    metrics["Mean-Prec"] = round(np.mean(precisions), 3)
    metrics["Mean-Recall"] = round(np.mean(recalls), 3)
    metrics["Mean-Dice"] = round(np.mean(dices), 3)

    # Calculate Mean2 metrics using total TP, FP, FN
    mean2_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0
    mean2_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0
    mean2_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0

    metrics["Mean2-Prec"] = round(mean2_precision, 3)
    metrics["Mean2-Recall"] = round(mean2_recall, 3)
    metrics["Mean2-Dice"] = round(mean2_dice, 3)
    metrics['Filename'] = filename

    # Create DataFrame from metrics dictionary
    df = pd.DataFrame([metrics])

    # Order columns
    ordered_cols = ['Filename'] + sorted([col for col in df.columns if col != 'Filename'], key=lambda x: (x.split('-')[0], x.split('-')[1]))
    df = df[ordered_cols]

    return df

# Note: This example assumes that 'output_array', 'gt_array', and 'filename' are defined. 
# Replace 'output_array', 'gt_array', "example_filename", and 'n_classes' with actual values to use this function.
# df = evaluate_segmentation_with_averages_updated(output_array, gt_array, "example_filename", n_classes=5)
# print(df)



def calculate_sizes(input_dir, filename, num_classes, verbose=True):
    
    # input_file = os.path.join(input_dir, filename + "_05mm_simulated.nii.gz")
    input_file = glob.glob(os.path.join(input_dir, filename + "_05mm_simulated*.nii.gz"))[0]

    img = nib.load(input_file) # Load the NIfTI file
    data = np.array(img.dataobj) # Get the image data as a numpy array

    sizes = []
    for i in range(0, num_classes): # Loop over all classes
        volume = np.sum(data == i)
        sizes.append(volume)
    sizes = np.array(sizes)
    if verbose:
        print(f"Sizes: {sizes}")

    sizes = [size  if size > 0 else 1 for idx, size in enumerate(sizes)]
    return sizes, data

def largest_connected_components(labels_3d):
    
    from skimage.measure import label, regionprops
    
    classes = np.unique(labels_3d)
    output = np.zeros_like(labels_3d)

    for c in classes:
        if c == 0:  # assuming 0 is the background
            continue
        binary_img = labels_3d == c
        labeled_img = label(binary_img)
        regions = regionprops(labeled_img)
        if len(regions) > 0:
            largest_component = max(regions, key=lambda r: r.area)
            output[labeled_img == largest_component.label] = c
    
    return output



def apply_closing_and_fill_holes_to_multiclass_3d_array(array_3d):
    # Get the unique classes in the array
    classes = np.unique(array_3d)

    # Initialize an array to store the result
    processed_array = np.zeros_like(array_3d)

    # Create an anisotropic structure element
    struct_element = np.zeros((5, 3, 3))
    struct_element[2, 1, 1] = 1  # Center element
    struct_element[1:4, 0:3, 0:3] = 1  # XY plane elements
    struct_element[0:5, 1, 1] = 1  # Z axis elements

    for c in classes:
        # Create a binary mask for the current class
        binary_mask = np.where(array_3d == c, 1, 0)

        # Apply the closing operation to the binary mask
        closed_mask = binary_closing(binary_mask, structure=struct_element)

        # Fill holes in the closed mask
        filled_mask = binary_fill_holes(closed_mask)

        # Add the processed mask for the current class to the result array
        processed_array = np.where(filled_mask, c, processed_array)

    return processed_array





def weighted_median(data, weights):
    """
    Compute the weighted median of a 1D numpy array.

    Arguments:
    data -- 1D numpy array
    weights -- 1D numpy array with the same size as data

    Returns:
    median -- float, weighted median of data
    """
    sorted_data, sorted_weights = map(np.array, zip(*sorted(zip(data, weights))))
    cum_weights = np.cumsum(sorted_weights)
    return sorted_data[(cum_weights - 0.5 * sorted_weights) >= 0.5 * np.sum(sorted_weights)][0]

def weighted_median_filter(volume, weights, footprint):
    """
    Apply a weighted median filter to a 3D array.

    Arguments:
    volume -- 3D array, input data
    weights -- 3D array, weights for the weighted median
    footprint -- 3D boolean array, specifies which neighboring elements to consider

    Returns:
    result -- 3D array, result of applying the filter
    """
    # Flatten the footprint, volume, and weight arrays
    fp = footprint.flatten()
    flat_vol = volume.flatten()
    flat_weights = weights.flatten()

    # Create an array for the output
    result = np.empty_like(flat_vol)

    # Apply the weighted median filter
    for i in range(len(flat_vol)):
        # Create array of element indices for the current footprint
        indices = np.flatnonzero(fp) + i - len(fp)//2

        # Clip indices so they don't go out of the bounds of the input arrays
        indices = np.clip(indices, 0, len(flat_vol)-1)

        # Gather elements and their corresponding weights
        elements = flat_vol[indices]
        element_weights = flat_weights[indices]

        # Compute the weighted median
        result[i] = weighted_median(elements, element_weights)

    # Reshape the result array back into the original shape of volume
    return result.reshape(volume.shape)

def postprocess(prediction, size_reference, window_size=3):
    '''
    Applies postprocessing on prediction, smoothening in Z direction.

    Arguments:
    prediction -- 3D array, output from UNet
    size_reference -- 3D array, reference for size comparison
    window_size -- int, size of the window to consider for smoothing

    Returns:
    result -- 3D array, postprocessed prediction
    '''
    # Calculate weights inversely proportional to the counts of each class
    counts = Counter(size_reference.flatten())
    epsilon = 1e-7  # small constant to avoid division by zero
    weights = np.array([1/(counts[i] + epsilon) for i in prediction.flatten()]).reshape(prediction.shape)

    # Generate a binary structure for the neighborhood
    footprint = generate_binary_structure(3, 1)
    footprint = np.expand_dims(footprint, axis=2)
    footprint = np.repeat(footprint, window_size, axis=2)

    # Apply the weighted median filter
    return weighted_median_filter(prediction, weights, footprint)


def process_file(file_path, model, modal, output_dir, device, args, sliding_window=False, patch_size=(256, 256), stride=(192, 192), num_classes=4, ideal_size_file = None):
    
    # load image
    img_nii = nib.load(file_path)
    img_nii = nib.as_closest_canonical(img_nii)
    img = np.array(img_nii.dataobj).astype(np.float32)
    out_affine = img_nii.affine

    if args.carrenD:
      img = (img + 4.0) / 8.5   # pre-process in "dataloader.py"
      img = img[::-1, ::-1, :]  # flip the image in the x and y axes to match training patches
    else:
      img = rotate_axis(img, args.plane)
      img = normalize_image(img, modal)

    img = np.copy(img)
    output_array = np.zeros(img.shape, dtype=np.uint8)
    # output_array_raw = np.zeros(([num_classes, img.shape[0], img.shape[1], img.shape[2]] ), dtype=np.float32)

    if sliding_window:
      from torch.nn.functional import softmax
      from scipy.ndimage import uniform_filter

      for sliceIdx in trange(img.shape[2], leave=False, desc=os.path.basename(file_path)):
          img_slice = torch.from_numpy(img[...,sliceIdx]).float()
          # Array to store the output results
          output_map = np.zeros((*img[...,sliceIdx].shape, num_classes+1), dtype=float)
          count_map = np.zeros(img[...,sliceIdx].shape, dtype=float)
          img_slice = img_slice.to(device)

          # Split the image with a sliding window
          for i in range(0, img_slice.shape[0] - patch_size[0] + 1, stride[0]):
              for j in range(0, img_slice.shape[1] - patch_size[1] + 1, stride[1]):
                  # Extract the patch
                  patch = img_slice[i:i + patch_size[0], j:j + patch_size[1]]
                  # Shape the patch according to the model input
                  patch = patch.unsqueeze(0).unsqueeze(0)  # e.g., if the model expects [batch, channel, height, width]

                  # Inference on the patch
                  with torch.no_grad():
                      logits = model(patch)
                      if type(logits) == list:
                          logits = logits[0]

                  # Convert the logits to probabilities with softmax
                  probabilities = softmax(logits, dim=1).squeeze().cpu().numpy()

                  # Add the results of the patch to the output map
                  output_map[i:i + patch_size[0], j:j + patch_size[1]] += probabilities.transpose((1, 2, 0))
                  count_map[i:i + patch_size[0], j:j + patch_size[1]] += 1
                  output_map[i:i + patch_size[0], j:j + patch_size[1]] /= count_map[i:i + patch_size[0], j:j + patch_size[1], np.newaxis]

          # Compute the class indices by argmax
          class_map = np.argmax(output_map, axis=-1)
          output_array[..., sliceIdx] = class_map 

    else:
      # pad to the nearest power of 2
      new_size = [int(2 ** np.ceil(np.log2(s))) for s in img.shape[:2]]
      img_tensor = F.pad(torch.tensor(img), (0, 0, 0, new_size[1]-img.shape[1], 0, new_size[0]-img.shape[0]))

      # infer and construct the output tensor
      # output_tensor = torch.zeros(img.shape) # torch.zeros(*new_size, img_tensor.shape[2])

      if ideal_size_file is not None:
        sizes, ideas_size_arr = calculate_sizes(ideal_size_file, os.path.basename(file_path).replace("_image","_label").replace(".nii.gz",""), num_classes)
        class_weights = 1.0 / np.array(sizes)
        print(f"Class weights: {class_weights}")

      for i in trange(img_tensor.shape[2], leave=False, desc=os.path.basename(file_path)):
          # move slice tensor to GPU and add channel and batch dimensions
          
          slice_tensor = img_tensor[:,:,i].unsqueeze(0).unsqueeze(0).to(device)
          # slice_tensor = (slice_tensor + 4.0) / 8.5
          # forward pass

          model_output = model(slice_tensor)
          
          if type(model_output) == list: # unet++ or other networks with deep-supervision
              model_output = model_output[0]

          with torch.no_grad():
            output_slice = model_output.squeeze(0)

          # Move the output to CPU
          output_slice = output_slice.cpu()
          
          # output_array_raw[..., i] = output_slice[:, :img.shape[0], :img.shape[1]].numpy()

          if ideal_size_file is not None:
            output_slice = output_slice * class_weights[:, None, None] # apply the weights

          output_slice = torch.argmax(output_slice, dim=0)
          output_array[:,:,i] = output_slice[:img.shape[0], :img.shape[1]].numpy()

    if not args.carrenD:
      output_array = rotate_axis(output_array, args.plane, inverse=True)

    # save output tensor to nifti file
    output_file = osp.join(output_dir, osp.basename(file_path).replace('.nii.gz', '_1pred.nii.gz'))
    nib.save(nib.Nifti1Image(output_array, out_affine), output_file)
    
    # nib.save(nib.Nifti1Image( (1000 * img).astype(np.int16), out_affine), output_file.replace('_pred.nii.gz', '_image.nii.gz'))
  
    
    # post-processing
    if args.postprocess:
      output_array = apply_closing_and_fill_holes_to_multiclass_3d_array(output_array)
      output_array = largest_connected_components(output_array)
      nib.save(nib.Nifti1Image(output_array, out_affine), output_file.replace("_1pred.nii.gz", "_1pred_post.nii.gz"))

    # open the ground-truth file if it exists
    label_path = file_path.replace('image', 'gth') if args.carrenD else file_path.replace('_image.nii.gz', '_label.nii.gz')
    if os.path.exists(label_path):
        label_nii = nib.load(label_path)
        label_nii = nib.as_closest_canonical(label_nii)
        label_arr = np.array(label_nii.dataobj).astype(np.uint8)
        label_arr = rotate_axis(label_arr, args.plane)
      
        if args.carrenD:
          label_arr = label_arr[::-1, ::-1, :]  # flip the image in the x and y axes to match training patches

        label_reso = label_nii.header.get_zooms()

        out_label_path = output_file.replace('_1pred.nii.gz', '_2gt.nii.gz')
        if osp.exists(out_label_path):
            os.remove(out_label_path)
        try:
            os.symlink(os.path.relpath(os.path.realpath(label_path), start=os.path.dirname(out_label_path)), out_label_path)
        except FileExistsError as exc:
            print(exc)
            pass


    # create symbolic links to the input image
    out_image_path = output_file.replace('_1pred.nii.gz', '_3image.nii.gz')
    if osp.exists(out_image_path):
        os.remove(out_image_path)
    try:
        os.symlink(os.path.relpath(os.path.realpath(file_path), start=os.path.dirname(out_image_path)), out_image_path)
    except FileExistsError as exc:
        print(exc)
        pass
    
    # evaluate
    if os.path.exists(label_path):
        eval = evaluate_segmentation(output_array, label_arr, filename=os.path.basename(file_path), n_classes=num_classes)
        return eval                          
    else:
        return None

    # create symbolic links
    '''
    symlink_image = osp.join(output_dir, osp.basename(file_path))
    symlink_label = symlink_image.replace('_image.nii.gz', '_label.nii.gz')
    real_image = osp.join(osp.dirname(file_path), osp.basename(file_path))
    real_label = real_image.replace('_image.nii.gz', '_label.nii.gz')

    if not osp.exists(symlink_image):
        os.symlink( os.path.normpath(osp.relpath(real_image, output_dir)), symlink_image)
    if not osp.exists(symlink_label):
        os.symlink( os.path.normpath(osp.relpath(real_label, output_dir)), symlink_label)
    '''

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for entire CT volumes.')

    parser.add_argument('-p', '--pkl', type=str, required=True,
                        help='Path to the pretrained model file.')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to the directory of input nifti files.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Path to the output directory.')
    parser.add_argument('-m', '--modal', type=str, default=None,
                        choices=['ct', 'mr'], 
                        help='Modality of the input image (ct or mr).')
    parser.add_argument("-pl", "--plane", default=None, choices=["axial","coronal","sagittal"], help="Axial, coronal or sagittal plane. Adjust to the trained model")
    parser.add_argument('-s', '--sizeref', type=str, default=None,
                        help='File for size reference.')
    parser.add_argument('-w', '--sliding_window', action='store_true')
    parser.add_argument('-pp', '--postprocess', action='store_true')
    parser.add_argument('-v', '--verbose', type=int, default=1)
    parser.add_argument("--carrenD", action="store_true", help="carrenD data available at https://github.com/carrenD/Medical-Cross-Modality-Domain-Adaptation")

    args = parser.parse_args()

    if args.modal is None:
        args.modal = "ct" if "ct_" in args.input else "mr"
        print("Modality not provided. Using default modality: {}".format(args.modal))

    if args.plane is None:
        if args.pkl.find("sagittal") > 0:
            args.plane = "sagittal"
        elif args.pkl.find("coronal") > 0:
            args.plane = "coronal"
        else:
            args.plane = "axial"
        print("Plane not provided. Using estimated plane: {}".format(args.plane))

    # Set output directory to be "entire" in the model file's directory if not provided
    if args.output is None:
        model_dir = os.path.dirname(args.pkl)
        args.output = os.path.join(model_dir, "entire_"+os.path.splitext(os.path.basename(args.pkl))[0])
        cprint(f'Output directory not provided. Using default directory: {args.output}', "yellow")
    if args.sizeref is not None:
        args.output = args.output + "_sizeref"

    os.makedirs(args.output, exist_ok=True)
    return args

def main():
    args = parse_args()

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.pkl)
    model = model.eval().to(device)

    # output directory
    os.makedirs(args.output, exist_ok=True)

    # get all input files
    input_files = sorted(glob.glob(osp.join(args.input, '*.nii.gz')))
    input_files = [f for f in input_files if f.find("gth")<0 and f.find("label")<0]

    pbar = tqdm(input_files, unit="file")

    scores = []

    for file_path in pbar:
        # pbar.set_description(f"Processing {file_path}")
        score = process_file(file_path, model, args.modal, args.output, device,args, ideal_size_file=args.sizeref, sliding_window=args.sliding_window,
                             patch_size=(256, 256) if args.carrenD else (224, 224),
                             num_classes=4 if args.carrenD else 7)
        scores.append(score)
    pbar.close()

    # save scores to csv
    scores = pd.concat(scores, ignore_index=True)
    
    scores = scores.dropna() # remove None from scores
    '''
    if len(scores) > 0:
      # Create rows for mean and standard deviation, with 'Filename' set to '(Mean)' and '(StdDev)', respectively
      mean_row = pd.Series(["(Mean)"] + list(scores.mean()), index=scores.columns)
      std_dev_row = pd.Series(["(StdDev)"] + list(scores.std()), index=scores.columns)
      min_row = pd.Series(["(Min)"] + list(scores.min())[1:], index=scores.columns)
      max_row = pd.Series(["(Max)"] + list(scores.max())[1:], index=scores.columns)

      # Append these rows to the DataFrame
      for row in [mean_row, std_dev_row, min_row, max_row]:
        scores = scores.append(row, ignore_index=True)
    '''
    scores.to_csv(osp.join(args.output, "scores.csv"), index=False)


if __name__ == "__main__":
    main()
