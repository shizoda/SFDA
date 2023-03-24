import sys, os, pdb, glob
import nibabel as nib
import argparse
import numpy as np
import tqdm

def parseArgs():
  parser = argparse.ArgumentParser(description='PNG files of SFDA input and output are converted as nii.gz files. This program is useful for observing results on MITK-Workbench or ITK-SNAP')
  parser.add_argument("-i", "--inputDir", type=str, required=True, help="Directory for input images")
  parser.add_argument("-g", "--gtDir", type=str, required=True, help="Directory for ground-truths")
  parser.add_argument("-p", "--predDir", type=str, required=True, help="Directory for predicted results")
  parser.add_argument("-n", "--nMax", type=int, default=300, help="Maximum number of slices per each output nii file. This is not strictly followed")
  parser.add_argument('-u', "--use_name", action='store_true', help="Prefix, e.g. CaseXX_... , are considered to generate results")
  return parser.parse_args()

def imagesTo3dArr(paths, dtype=np.float32):
  if paths[0].find(".nii")>=0:
    sampleImg = np.array(nib.load(paths[0]).dataobj)
    outArr = np.zeros((sampleImg.shape[1], sampleImg.shape[2], len(paths)), dtype=dtype)
  else:
    from skimage import io
    sampleImg = io.imread(paths[0])
    outArr = np.zeros((sampleImg.shape[0], sampleImg.shape[1], len(paths)), dtype=dtype)
  for idx in range(len(paths)):
    if paths[idx].find(".nii")>=0:
      outArr[..., idx] = np.array(nib.load(paths[idx]).dataobj)[0, ...]
    else:
      from skimage import io
      outArr[..., idx] = io.imread(paths[idx])
  return outArr

def getPaths(dirPath):
  import re
  def atoi(text):
      return int(text) if text.isdigit() else text
  def natural_keys(text):
      return [ atoi(c) for c in re.split(r'(\d+)', text) ]
  return [path for path in sorted(glob.glob(os.path.join(dirPath, "*")), key=natural_keys) if path.find("_0.nii")>=0 or path.find("_0.png")>=0]

def groupWithPrefix(paths, outDir, suffix, nMax, useName=False, dtype=np.int16):
   names = [os.path.basename(path) for path in paths]
   if useName:
    prefixes = [name.split("_")[0] for name in names]
    prefixesUnique = sorted(set(prefixes), key=prefixes.index)

   print("Loaded", suffix)

   if len(paths)==0:
     print("No paths!", suffix)
     pdb.set_trace()

   for prefix in (prefixesUnique if useName else ["_"]):
      targetPathsAll = [path for path in paths if path.find(prefix)>=0]
      nDivs = max(round( len(targetPathsAll) / nMax ), 1)
      targetPathsGrouped = np.array_split(targetPathsAll, nDivs)

      for groupIdx, targetPaths in tqdm.tqdm(enumerate(targetPathsGrouped), total=len(targetPathsGrouped)):
        out3dArr = imagesTo3dArr(targetPaths.tolist(), dtype=dtype)

        if useName:
          outPath = os.path.join(outDir, prefix + "-" + suffix + ".nii.gz")
        else:
          prefix2 = ("grp" + str(groupIdx) if len(targetPathsGrouped)>1 else "")
          outPath = os.path.join(outDir, prefix2 + "-" + suffix + ".nii.gz")
        
        out3dArr = out3dArr.transpose((1,0,2))
        nib.save(nib.Nifti1Image(out3dArr, np.diag((-1,-1,-1,1))), outPath)

if __name__=="__main__":
    args = parseArgs()
    inputPaths = getPaths(args.inputDir)
    gtPaths = getPaths(args.gtDir)
    predPaths = getPaths(args.predDir)

    outDir = os.path.join(args.predDir, "nii")
    os.makedirs(outDir, exist_ok=True)

    groupWithPrefix(predPaths, outDir, "pred", args.nMax, args.use_name, dtype=np.uint8)
    groupWithPrefix(inputPaths, outDir, "img", args.nMax, args.use_name, dtype=np.float32)
    groupWithPrefix(gtPaths, outDir, "gt", args.nMax, args.use_name, dtype=np.uint8)
