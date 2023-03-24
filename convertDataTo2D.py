import pdb, os, sys, glob
import nibabel as nib
import numpy as np
import tqdm
import pandas as pd

classIdxs = [0,1,2,3,4]
nClasses  = len(classIdxs)

debug = False
if debug:
  print("WARNING: debug mode")

def meanWithoutZero(vals):
  out = []
  for col in range(vals.shape[1]):
    valsWithoutZero = [val for val in vals[:,col].tolist() if val > 0.01]
    if len(valsWithoutZero)>0:
      out.append(int(np.round(np.mean(valsWithoutZero))))
    else:
      out.append(0)
  return out

def mapMMWHS(inArr):
  outArr = np.zeros_like(inArr, dtype=np.uint8)
  outArr[inArr==500] = 1 # LV cavity
  outArr[inArr==600] = 2 # RV cavity
  outArr[inArr==420] = 3 # LA cavity
  outArr[inArr==550] = 4 # RA cavity
  outArr[inArr==205] = 5 # Myocardium of LV
  outArr[inArr==820] = 6 # Ascending Aorta
  outArr[inArr==850] = 7 # Pulmonary Artery
  return outArr

def addEstimatedSizes(df, meanGtSizes):
  withTagsColumn = []
  nRows = len(df)

  for rowIdx in range(nRows):
    gtSizes = df.iloc[rowIdx]["val_gt_size"]

    oneOut = [int(round(meanGtSizes[sizeIdx])) if gtSizes[sizeIdx]>0 else 0 for sizeIdx in range(nClasses)]
    withTagsColumn.append(oneOut)
    
  df["dumbprednotags"] = [list(meanGtSizes) for idx in range(nRows)]  # no tagging information
  df["dumbpredwtags"]  = withTagsColumn

  return df

if __name__=="__main__":
  modality = ["ct", "mr"]
  purpose  = ["train", "val"]
  defaultAffine = np.diag((1,1,1,1))

  dfs = []

  for mod in modality:
    df = pd.DataFrame(index=[], columns=["val_ids", "modality", "purpose", "val_gt_size", "dumbprednotags", "dumbpredwtags"])
    prev=0
    for purp in purpose:
      # prefix="ctslice" if purp=="train" else "valctslice"

      inDir = os.path.join("dataOld", mod, purp)
      outDir = os.path.join("./data_mmwhs", mod, purp)

      os.makedirs(os.path.join(outDir, "GT"),  exist_ok=True)
      os.makedirs(os.path.join(outDir, "IMG"), exist_ok=True)
      
      files = [os.path.basename(path) for path in sorted(glob.glob(os.path.join(inDir, "GT", "*.nii.gz") )) ]
      
      for idx, fileName in tqdm.tqdm(enumerate(files)):
        gtArr  = np.array(nib.load(os.path.join(inDir, "GT",  fileName )).dataobj)
        # gtArr  = mapMMWHS(gtArr)
        imgArr = np.array(nib.load(os.path.join(inDir, "IMG", fileName )).dataobj)

        gtSizes = [np.sum(gtArr==val) for val in classIdxs]
        
        for idxZ in range(imgArr.shape[2]):
          prefix = ("val" if purp=="val" else "") + mod + "slice"
          outName = prefix + os.path.basename(fileName).replace(".nii","").replace(".gz","") + "_" + str(idxZ+1) + ".nii"

          nib.save(nib.Nifti1Image(gtArr[..., 0][np.newaxis, ...].astype(np.uint8), defaultAffine), os.path.join(outDir, "GT", outName ))
          nib.save(nib.Nifti1Image(imgArr[..., idxZ][np.newaxis, ...], defaultAffine), os.path.join(outDir, "IMG",  outName))

          record = pd.Series([os.path.basename(outName), mod, purp, gtSizes, -1, -1], index=df.columns)
          df = df.append(record, ignore_index=True)
        
        if debug and idx>30:
<<<<<<< HEAD
          break 
=======
          break
>>>>>>> 49610e6b94345c227ba7a8d8132746a1b6b1b5e2
  
    # Average for training data of each modality
    sset = df[df["modality"]==mod][df["purpose"]=="train"]
    allGtSizes = np.array(tuple(sset["val_gt_size"]))
    meanGtSizes = meanWithoutZero(allGtSizes)

    df = addEstimatedSizes(df, meanGtSizes)
    dfs.append(df)

  df = pd.concat(dfs)
  df.to_csv("/home/hoda/git/SFDA/sizes/whs_simple.csv", index=False)
