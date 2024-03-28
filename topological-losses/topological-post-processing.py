from importlib import reload
import numpy as np
import torch
import nibabel as nib
from tqdm import trange
from topo import get_differentiable_barcode, multi_class_topological_post_processing
import pdb

import os, sys; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
device = torch.device('cuda:0')


def unique_with_min_proportion(arr, min_proportion):
    unique, counts = np.unique(arr, return_counts=True)
    proportion = counts / len(arr)
    mask = proportion >= min_proportion
    return unique[mask]

# ## Test time adaptation by topological post-processing

# In[36]:


class TopologicalPostProcessing:
    
    # shared dict
    trained_models = {}
    
    def __init__(self, model, lr=1e-4, mse_lambda=1000, opt=torch.optim.Adam,
                 num_its=100, construction='0', thresh=0.5, parallel=False):
        
        if type(model) == str:
            # Load the model from the specified path
            self.model = torch.load(model).to(device)
        else:
            self.model = model.to(device)

        self.lr = lr
        self.mse_lambda = mse_lambda
        self.opt = opt
        self.num_its = num_its
        self.construction = construction
        self.thresh = thresh
        self.parallel = parallel

    def gen_prior(self, classes):
        # Generate a new dictionary based on the 'prior' dictionary, but only include the keys that are present in the 'classes' argument.

        # Define the 'prior' dictionary
        prior = {
            (1,): (1, 0),
            (2,): (1, 0),
            (3,): (1, 0),
            (4,): (1, 0),
            (5,): (1, 0),
            (6,): (1, 0),
            (7,): (1, 0),
            (3, 1): (1, 0),
            (1, 5): (1, 0),
            (5, 2): (1, 0),
            (2, 7): (1, 0),
            (2, 4): (1, 0),
            (1, 6): (1, 0),
        }

        # Create a new dictionary to store the filtered entries
        new_prior = {}

        # Iterate over the items in the 'prior' dictionary
        for key, value in prior.items():
            # Check if all elements of the key are in 'classes'
            if all(element in classes for element in key):
                # If so, add the key-value pair to the 'new_prior' dictionary
                new_prior[key] = value

        # Return the 'new_prior' dictionary
        return new_prior

    def process_image(self, image, classes = None):

      if classes is None:
          pred_unet = torch.softmax(model(image), 1).squeeze().argmax(0).cpu().numpy()
          classes = unique_with_min_proportion(pred_unet, 0.05)
          classes = classes[classes!=0]
      classes = tuple(sorted(classes))

      prior = self.gen_prior(classes)

      # Find the model for the given classes
      
      if classes in TopologicalPostProcessing.trained_models:
          model_TP = TopologicalPostProcessing.trained_models[classes]
      else:
          # Train the model for the given classes
          model_TP = multi_class_topological_post_processing(
              inputs=image, model=self.model, prior=prior,
              lr=self.lr, mse_lambda=self.mse_lambda,
              opt=self.opt, num_its=self.num_its, construction=self.construction, thresh=self.thresh, parallel=self.parallel
          )
          TopologicalPostProcessing.trained_models[classes] = model_TP
          
      pred_topo = torch.softmax(model_TP(image), 1).squeeze().argmax(0).cpu().numpy()
      return pred_topo


def process_nii(nii_path, model):
    
    # Load nii file as numpy array
    nii = nib.load(nii_path)
    image_arr = np.array(nii.dataobj)
    image_arr = image_arr.astype(np.float32) / 10000.0
    out_arr = np.zeros_like(image_arr, dtype = np.uint8)

    for idx in trange(image_arr.shape[2]):
        
        image = image_arr[..., idx:idx+1]
        image = np.transpose(image, (2,0,1))[np.newaxis,...]
        uniq = np.unique(image)
        image = torch.tensor(image).to(device)

        try:
          # Run topological post-processing
          topo = TopologicalPostProcessing(model)
          pred_topo = topo.process_image(image)
          out_arr[..., idx] = pred_topo
        except Exception as e:
          print(f"Error processing image {idx}: {e}")
          pred_unet = torch.softmax(model(image), 1).squeeze().argmax(0).cpu().numpy()
          out_arr[..., idx] = pred_unet
          continue

    # Save the processed nii file
    pred_topo_nii = nib.Nifti1Image(out_arr, affine=nii.affine)
    nib.save(pred_topo_nii, nii_path.replace(".nii.gz", "_topo.nii.gz"))
      
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(description="Processes *_1pred.nii.gz files in the specified directory.")
    parser.add_argument('-m', '--model', required=True, help='Path to the model')
    parser.add_argument('-i', '--input', required=True, help='Path to the directory containing input nii files')
    args = parser.parse_args()

    model = torch.load(args.model)
    process_nii(args.input, model)