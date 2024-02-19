from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import os, sys, pdb, argparse
from tqdm import tqdm, trange

def get_mask_generator(model_type="vit_h", sam_checkpoint="sam_vit_h_4b8939.pth", device = "cuda"):
  sam = sam_model_registry[model_type](checkpoint=os.path.join(os.path.dirname(__file__), "model_sam", sam_checkpoint))
  sam.to(device=device)
  mask_generator = SamAutomaticMaskGenerator(sam)
  return sam, mask_generator



def get_labeled_image(masks, return_stack=False, max_labels=5):
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
            stack[:, :, i] = (mask['segmentation'].astype(np.uint8) > 0).astype(stack.dtype)
        return stack
    else:
        # Initialize the 2D label image with zeros
        labels = np.zeros(sorted_masks[0]['segmentation'].shape, dtype=(np.uint8 if max_labels<256 else np.uint16))
        for i, mask in enumerate(sorted_masks):
            labels[mask['segmentation']] = i + 1
        return labels


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def convert_gray_to_bgr(npimg):
    img_scaled = np.clip(npimg, 0.0, 1.0)
    bgr_img = np.zeros((*img_scaled.shape, 3), dtype=np.uint8)
    bgr_img[..., 0] = (255 * img_scaled).astype(np.uint8)
    bgr_img[..., 1] = (255 * sigmoid(10.0 * img_scaled - 5.0)).astype(np.uint8)
    bgr_img[..., 2] = 255 * np.clip(img_scaled, 0.25, 0.75) 
    # bgr_img = np.stack([img_scaled] * 3, axis=-1)
    return bgr_img


def get_3d_image_mask(input_img, mask_generator, max_labels=5):

  out_img = np.zeros_like(input_img, dtype=(np.uint8 if max_labels<256 else np.uint16))
  for z in trange(input_img.shape[2]):
    bgrimg = convert_gray_to_bgr(input_img[..., z])
    masks = mask_generator.generate(bgrimg)
    out_img[..., z] = get_labeled_image(masks, return_stack=False, max_labels=max_labels)
  return out_img

def predict_3d_image_seg(input_img, predictor,
                         coords=[[96, 96], [160, 160]], labels=[1, 2],
                         multimask_output=False,
                         input_boxes=None):
  
  out_img = np.zeros_like(input_img, dtype=np.uint8)
  for z in trange(input_img.shape[2]):
    bgrimg = convert_gray_to_bgr(input_img[..., z])
    predictor.set_image(bgrimg)
    masks, scores, logits = predictor.predict(
      point_coords=np.array(coords),
      point_labels=np.array(labels),
      input_boxes=input_boxes,
      multimask_output=multimask_output,
    ) 
    
    pdb.set_trace()
    for lbl in range(masks.shape[0]-1, -1, -1):
      out_img[masks[lbl,...],z] = lbl+1

    if z==10:
       break
    
  return out_img


if __name__=="__main__":
  sam, mask_generator = get_mask_generator()
  predictor = SamPredictor(sam)
  print("Mask generator and predictor loaded.")

  argparse = argparse.ArgumentParser(description="Segment a 3D image using the SAM model.")
  argparse.add_argument("-i", "--input_file", required=True, help="Input file path")
  argparse.add_argument("-o", "--output_file", required=True, help="Output file path")
  argparse.add_argument("-m", "--max", default=1, type=float, help="Maximum value for the input image")
  argparse.add_argument("-n", "--nlabels", default=100, type=float, help="Maximum number of labels")
  args = argparse.parse_args()

  import nibabel as nib
  input_nii = nib.load(args.input_file)
  input_img = np.array(input_nii.dataobj) / args.max

  input_box = np.zeros_like(input_img[...,0], dtype=np.uint8)[np.newaxis,...]
  input_box = int(0.3 * input_box.shape[0]) : int(0.7 * input_box.shape[0]) 

  out_seg = predict_3d_image_seg(input_img, predictor, mask_input=[input_box])
  nib.save(nib.Nifti1Image(out_seg, input_nii.affine), args.output_file)

  out_img = get_3d_image_mask(input_img, mask_generator, max_labels=args.nlabels)
  nib.save(nib.Nifti1Image(out_img, input_nii.affine), args.output_file.replace(".nii", "_mask.nii"))



