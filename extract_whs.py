import os, pdb
import numpy as np
import nibabel as nib
from extract_misc import *
import SimpleITK as sitk
import pandas as pd
import ast
from tqdm import tqdm, trange
from termcolor import colored, cprint

def try_literal_eval(x):
    try:
        return ast.literal_eval(x)
    except ValueError:
        return x  # if ast.literal_eval fails, return the original value

def inverse_size(df_A, df_B):
    # Convert the string lists to literal lists
    for column in ['dumbpredwtags', 'dumbprednotags']:
        df_A[column] = df_A[column].apply(ast.literal_eval)
        df_B[column] = df_B[column].apply(ast.literal_eval)

    # Calculate the maximum of each position in lists across the dataframe for both dataframes
    max_values_B = [max([row[i] for row in df_B['dumbprednotags']]) for i in range(len(df_B['dumbprednotags'].iloc[0]))]

    # Swap non-zero values of each element in the list with the corresponding maximum value
    df_A['dumbpredwtags'] = df_A['dumbpredwtags'].apply(lambda row: [x if x == 0 else max_values_B[i] for i, x in enumerate(row)])
    df_A['dumbprednotags'] = df_A['dumbprednotags'].apply(lambda row: [x if x == 0 else max_values_B[i] for i, x in enumerate(row)])

    max_values_A = [max([row[i] for row in df_A['dumbprednotags']]) for i in range(len(df_A['dumbprednotags'].iloc[0]))]

    # Swap non-zero values of each element in the list with the corresponding maximum value
    df_B['dumbpredwtags'] = df_B['dumbpredwtags'].apply(lambda row: [x if x == 0 else max_values_A[i] for i, x in enumerate(row)])
    df_B['dumbprednotags'] = df_B['dumbprednotags'].apply(lambda row: [x if x == 0 else max_values_A[i] for i, x in enumerate(row)])

    return df_A, df_B


if __name__=="__main__":

  # argparse
  import argparse
  parser = argparse.ArgumentParser(description='Extract WHS dataset')
  parser.add_argument('-i', '--input_dir', type=str, default='mmwhs_orig', help='input directory')
  parser.add_argument('-is','--input_dir_suffix', type=str, default='_075x075x075/cropped', help='input directory suffix')
  parser.add_argument('-o', '--output_dir_prefix', type=str, default='patches/data_whs', help='prefix of output directory')
  parser.add_argument('-r', '--resolution', type=float, nargs=3, default=[0.75, 0.75, 0.75], help='resolution')
  parser.add_argument('-c', '--simct', type=str, default=None, help="simulated label directory for CT")
  parser.add_argument('-m', '--simmr', type=str, default=None, help="simulated label directory for MR")
  parser.add_argument('-n', '--n_folds', type=int, default=10, help='number of folds')
  parser.add_argument("-O", "--overlaps", action='store_true', help="Use overlaps")
  parser.add_argument("-s", "--patch_size", type=int, nargs=2, default=[224, 224], help="Patch size")
  parser.add_argument("-p", "--plane", nargs="*", default=["axial", "coronal", "sagittal"], choices=["axial","coronal","sagittal"], help="Plane")
  
  parser.add_argument('--debug', action='store_true')
  args = parser.parse_args()

  for plane in args.plane:
    output_dir = args.output_dir_prefix + args.input_dir_suffix.replace("/","_").replace("cropped", "cr") + "-"+ str(args.patch_size[0]) + "x" + str(args.patch_size[1])

    
    cprint(plane, "green")

    if args.simct is None and args.simmr is None:
      out_base_dir_suffix = "_" + plane
    elif args.simct is not None and args.simmr is not None:
      print("Both simct and simmr are specified. Use only one of them.")
      raise NotImplementedError()
    elif args.simct is not None:
      out_base_dir_suffix =  "_simct"
    elif  args.simmr is not None:
      out_base_dir_suffix = "_simmr"
    else:
      raise NotImplementedError()
    csv_suffix = out_base_dir_suffix

    csv_paths = {}

    for modal in [ "ct", "mr"]:

      input_dir = os.path.join(args.input_dir, modal + '_train' + args.input_dir_suffix)
      sim_dir = args.simct if modal == "ct" else args.simmr

      # Get list of image files and divide into train/test data
      img_files = [f for f in os.listdir(input_dir) if f.endswith('_image.nii.gz')]
      img_files.sort()

      np.random.seed(0)
      np.random.shuffle(img_files)

      if args.debug:
        print("Debug mode: using only 3 images")
        img_files = img_files[:3]
        args.n_folds = 3

      folds = split_img_paths(img_files, n=args.n_folds)
      if args.debug:
        print(folds[0][0])
        print(folds[0][1])

      for fold_idx, (train_files, test_files) in enumerate(folds):
        if fold_idx > 0:
          print("Skipping some dividing patterns because domain adaptation experiment does not need many patterns.")
          break

        df_train = None
        for mode in ("train", "test"):
          targeted_files = train_files if mode=="train" else test_files
          import itertools
          targeted_files_all = [(path, path.replace("_image.nii.gz", "_label.nii.gz")) for path in targeted_files ]

          df, out_csv_path = extract( input_dir, targeted_files, modal, mode, out_base_dir=output_dir + "_" + str(fold_idx+1) + out_base_dir_suffix, df_train = df_train, fold=fold_idx+1, sim_dir=sim_dir, debug = args.debug, csv_suffix = out_base_dir_suffix, overlap_dir=((args.simct if args.simct is not None else args.simmr) if args.overlaps else None), perform_save = True, patch_size=args.patch_size, resolution=args.resolution, plane=plane) 

          if mode=="train":
            df_train = df
            csv_paths[modal + str(fold_idx+1)] = out_csv_path   # common for "test" mode

    for fold_idx, (train_files, test_files) in enumerate(folds):
      if fold_idx > 0:
        print("Skipping some dividing patterns because domain adaptation experiment does not need many patterns.")
        break

      ct_size_path = csv_paths["ct" + str(fold_idx+1)]
      mr_size_path = csv_paths["mr" + str(fold_idx+1)]

      df_ct = pd.read_csv(ct_size_path, sep=";")
      df_mr = pd.read_csv(mr_size_path, sep=";")

      # update dataframes
      df_ct, df_mr = inverse_size(df_ct, df_mr)

      # write updated dataframes
      df_ct.to_csv(ct_size_path.replace('.csv', '_inv.csv'), index=False, sep=";")
      df_mr.to_csv(mr_size_path.replace('.csv', '_inv.csv'), index=False, sep=";")
