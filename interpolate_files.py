import os
import argparse
import nibabel as nib
import numpy as np
# from scipy.ndimage import zoom
import cupy as cp
from cupyx.scipy.ndimage import zoom
from tqdm import tqdm
from extract_misc import value_mapping


def calculate_pad_width(cropped_shape, crop_size):
    """Calculate padding width for each axis to match the target crop size."""
    pad_width = []
    for dim, cs in zip(cropped_shape, crop_size):
        if cs < 0:  # Skip padding for axes with negative crop size
            pad_width.append((0, 0))
            continue
        total_pad = cs - dim
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))
    return pad_width


def adjust_center_for_crop(c, half_size, length):
    """
    Adjust the center 'c' for cropping to ensure the crop area stays within the image boundaries.
    """
    if c - half_size < 0:
        c = half_size
    elif c + half_size > length:
        c = length - half_size
    return int(c)




def crop_image_with_padding(image, label, crop_size, method='centroid', important=[1,2,3,4,5], modal="ct"):
    """
    Crop the image around the specified labels' centroid or to minimize margin, and pad if the cropped size is smaller than crop_size.

    Parameters:
    - image: np.array, the 3D image to be cropped.
    - label: np.array, the 3D label indicating the region of interest.
    - crop_size: list or tuple, the size of the crop for each axis (x, y, z). Use negative value to skip cropping on that axis.
    - method: str, 'centroid' for centroid-based cropping or 'minimize_margin' for margin minimization.
    - important: list, labels considered important to be covered in the cropped image.

    Returns:
    - cropped_image: np.array, the cropped (and possibly padded) 3D image.
    - cropped_label: np.array, the cropped (and possibly padded) 3D label.
    """
    assert image.shape == label.shape, "Image and label must have the same shape."
    
    # Create a mask for the important labels
    important_mask = np.isin(label, important)

    if method == 'centroid':
        # Calculate the centroid of the important labels
        x, y, z  = np.where(important_mask)
        if not x.size or not y.size or not z.size:  # If no important labels are found, fall back to using all labels
            x, y, z = np.where(label > 0)
            import pdb; pdb.set_trace()
        centroid = np.mean(x), np.mean(y), np.mean(z)
        coords = centroid
    elif method == 'minimize_margin':
        # Calculate the bounding box of the important labels
        x, y, z  = np.where(important_mask)
        if not x.size or not y.size or not z.size:  # If no important labels are found, fall back to using all labels
            x, y, z  = np.where(label > 0)
            import pdb; pdb.set_trace()
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()
        coords = ((x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2)
    else:
        raise ValueError("Invalid method specified. Use 'centroid' or 'minimize_margin'.")

    # Calculate the start and end indices for cropping
    start, end = [], []

    for c, size, length in zip(coords, crop_size, image.shape):
        if size < 0:
            start.append(0)
            end.append(length)
        else:
            half_size = size // 2
            # Adjust the center 'c' to ensure the crop is within the image
            c_adjusted = adjust_center_for_crop(c, half_size, length)
            start.append(max(c_adjusted - half_size, 0))
            end.append(min(c_adjusted + half_size, length))

    # Crop the image and label
    cropped_image = image[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    cropped_label = label[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    print(coords, start, end)
    print(cropped_image.shape, end=" --> ")


    # Calculate padding if necessary
    cval = -1000 if modal == "ct" else 0  # Use -1000 for CT and 0 for MR
    pad_width = calculate_pad_width(cropped_image.shape, crop_size)
    cropped_image = np.pad(cropped_image, pad_width, mode='constant', constant_values=cval)
    cropped_label = np.pad(cropped_label, pad_width, mode='constant', constant_values=0)
    print(cropped_image.shape)

    return cropped_image, cropped_label




def resample_image(image, new_resolution, interpolation_method):

    original_data = np.asanyarray(image.dataobj)
    original_resolution = np.array(image.header.get_zooms())
    new_resolution = np.array(new_resolution)

    resample_ratio = original_resolution / new_resolution
    # resampled_data = zoom(original_data, resample_ratio, order=interpolation_method)
    resampled_data =  cp.asnumpy(zoom(cp.array(original_data, dtype=(np.float64 if interpolation_method != 0 else np.int32)), resample_ratio, order=interpolation_method))

    if interpolation_method == 0:
      if np.max(resampled_data) > 10:
        print("Value mapping performed.")
        resampled_data = value_mapping(resampled_data).astype(np.uint8)
        print("Now resampled_data contains", np.unique(resampled_data))
      else:
        resampled_data = resampled_data.astype(np.uint8)
    else:
      resampled_data = resampled_data.astype(image.dataobj.dtype)

    # Create a simple affine matrix with resolution information
    new_affine = np.eye(4)
    new_affine[:3, :3] = np.diag(new_resolution)
    new_affine [:, 3]  = 1

    return resampled_data, new_affine


def main():
    parser = argparse.ArgumentParser(description="Resample Nifti images.")
    parser.add_argument('-i', '--input', default="./mmwhs_orig/ct_train", help="Input directory containing nifti files.")
    parser.add_argument('-o', '--output', default=None, help="Output directory to save resampled nifti files.")
    parser.add_argument('-r', '--resolution', nargs=3, type=float, default=[0.75, 0.75, 0.75], help="Resolution after resampling.")
    parser.add_argument('-s', '--crop_size', type=float, default=[288, 288, 288])
    parser.add_argument("-m", "--crop_method", type=str, default="centroid", help="Method for cropping the image.")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), os.path.basename(args.input) + "_" + (str(args.resolution[0]) + "x" + str(args.resolution[1]) + "x" + str(args.resolution[2]) ).replace(".", ""))
        print(f"Output directory not specified. Saving resampled files to {args.output}")

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    nifti_files = [f for f in os.listdir(args.input) if f.endswith('_label.nii.gz')]

    for file_name in tqdm(nifti_files, desc="Processing files"):
        input_file = os.path.join(args.input, file_name)

        label = nib.load(input_file)
        label = nib.as_closest_canonical(label)

        image = nib.load(input_file.replace("label", "image"))
        image = nib.as_closest_canonical(image)

        # label
        label_arr, new_affine = resample_image(label,  args.resolution, interpolation_method=0)

        # image
        image_arr, _ = resample_image(image, args.resolution, interpolation_method=3)

        nib.save(nib.Nifti1Image(image_arr, new_affine), os.path.join(args.output, file_name.replace("label", "image")))
        nib.save(nib.Nifti1Image(label_arr, new_affine), os.path.join(args.output, file_name))

        # Crop the image and label
        os.makedirs(os.path.join(args.output, "cropped"), exist_ok=True)
        image_arr, label_arr = crop_image_with_padding(image_arr, label_arr, args.crop_size, method=args.crop_method, modal="ct" if "ct" in args.input else "mr")

        nib.save(nib.Nifti1Image(image_arr, new_affine), os.path.join(args.output, "cropped", file_name.replace("label", "image")))
        nib.save(nib.Nifti1Image(label_arr, new_affine), os.path.join(args.output, "cropped", file_name))


if __name__ == "__main__":
    main()
