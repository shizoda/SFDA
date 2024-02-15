from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import os, sys, pdb, argparse
from tqdm import tqdm, trange

def get_mask_generator(model_type="vit_h", sam_checkpoint="sam_vit_h_4b8939.pth", device = "cuda"):
  sam = sam_model_registry[model_type](checkpoint=os.path.join(os.path.dirname(__file__), "model_sam", sam_checkpoint))
  sam.to(device=device)
  mask_generator = SamAutomaticMaskGenerator(sam)
  return mask_generator

def convert_gray_to_bgr(npimg):
    img_scaled = np.clip(npimg * 255.0, 0, 255).astype(np.uint8)
    bgr_img = np.stack([img_scaled] * 3, axis=-1)
    return bgr_img


def get_labeled_image(masks, return_stack=False, max_labels=3):
    """
    Process annotations and return labeled images.
    
    Parameters:
    - anns: List of annotations, where each annotation contains a 'segmentation' field as a 2D bool np.array.
    - return_stack: If False, return a 2D array with labels. If True, return a stacked 3D array.
    - max_labels: Maximum number of labels to process. Smaller labels by area are ignored.
    
    Returns:
    - 2D or 3D np.array with labeled segments.
    """
    if len(masks) == 0:
        return None
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    
    if return_stack:
        # Initialize the stack with zeros
        stack = np.zeros((*sorted_masks[0]['segmentation'].shape, len(sorted_masks)), dtype=np.uint8)
        for i, mask in enumerate(sorted_masks):
            stack[:, :, i] = mask['segmentation'].astype(np.uint8) * (i + 1)
        return stack
    else:
        # Initialize the 2D label image with zeros
        labels = np.zeros(sorted_masks[0]['segmentation'].shape, dtype=np.uint8)
        for i, mask in enumerate(sorted_masks):
            labels[mask['segmentation']] = i + 1
        return labels

def segment_3d_image(input_img, mask_generator, max_labels=5):

  out_img = np.zeros_like(input_img, dtype=np.uint8)
  for z in trange(input_img.shape[2]):
    bgrimg = convert_gray_to_bgr(input_img[..., z])
    masks = mask_generator.generate(bgrimg)
    out_img[..., z] = get_labeled_image(masks, return_stack=False, max_labels=max_labels)
    
  return out_img


if __name__=="__main__":
  mask_generator = get_mask_generator()
  print("Mask generator loaded.")

  argparse = argparse.ArgumentParser(description="Segment a 3D image using the SAM model.")
  argparse.add_argument("-i", "--input_file", required=True, help="Input file path")
  argparse.add_argument("-o", "--output_file", required=True, help="Output file path")
  argparse.add_argument("-m", "--max", default=1, type=float, help="Maximum value for the input image")
  argparse.add_argument("-n", "--nlabels", default=100, type=float, help="Maximum number of labels")
  args = argparse.parse_args()

  import nibabel as nib
  input_nii = nib.load(args.input_file)
  input_img = np.array(input_nii.dataobj) / args.max
  out_img = segment_3d_image(input_img, mask_generator, max_labels=args.nlabels)
  nib.save(nib.Nifti1Image(out_img, input_nii.affine), args.output_file)
