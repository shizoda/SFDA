import argparse
import os
import h5py
import nibabel as nib
from tqdm import tqdm
import numpy as np

def convert_nii_to_h5_with_compression(input_dir, output_file, compression=None, preview=100):
    # 初期化
    preview_images = []
    out_affine = None

    with h5py.File(output_file, 'w') as h5f:
        nii_files = [f for f in os.listdir(input_dir) if f.endswith('.nii')]
        n_files = len(nii_files)
        step = max(1, n_files // preview)  # 等間隔に選ぶためのステップサイズを計算
        

        for i, nii_file in enumerate(tqdm(nii_files, desc="Converting NIfTI files")):
            file_path = os.path.join(input_dir, nii_file)
            image = nib.load(file_path)
            image_data = np.array(image.dataobj)
            
            if out_affine is None:
              out_affine = image.affine

            if input_dir.find("GT") >= 0:
                image_data = image_data.astype(np.uint8)
                compression = "lzf"

            h5f.create_dataset(os.path.basename(nii_file), data=image_data, compression=compression)
            
            # preview用の画像を選ぶ
            if i % step == 0:
                preview_images.append(image_data)

        # preview画像の処理
        if preview_images:
            # 画像サイズを取得
            image_shape = preview_images[0].shape
            # XY平面を持つ3D arrayを作成 (Z軸はpreviewの数)
            preview_stack = np.zeros((image_shape[1], image_shape[2], len(preview_images)), dtype=preview_images[0].dtype)
            for i, img in enumerate(preview_images):
                preview_stack[:, :, i] = img[0, :, :]  # 1チャンネル目を取得
            
            # NIfTIファイルとして保存
            preview_nii = nib.Nifti1Image(preview_stack, out_affine)
            preview_path = os.path.join(os.path.dirname(os.path.normpath(input_dir)), os.path.basename(os.path.normpath(input_dir)) + "_preview.nii.gz")
            nib.save(preview_nii, preview_path)
            print("Preview NIfTI file saved as:", preview_path)

def main():
    parser = argparse.ArgumentParser(description="Convert NIfTI files to a single HDF5 file.")
    parser.add_argument("-i", "--input_dir", required=True, help="Input directory containing NIfTI files")
    args = parser.parse_args()

    output_file = os.path.join(args.input_dir, "allfiles.h5")
    convert_nii_to_h5_with_compression(args.input_dir, output_file)
    print("Conversion completed! HDF5 file saved as:", output_file)

if __name__ == "__main__":
    main()
