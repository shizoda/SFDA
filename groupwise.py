import ants, groupwise, os, numpy as np, argparse, glob
from tempfile import mktemp
import nibabel as nib

# ANTspyの関数を直接インポート
from ants import registration, apply_transforms, resample_image_to_target
from ants.core import ants_image_io as iio
from ants.core import ants_transform_io as tio
from ants import utils
from tqdm import trange, tqdm

import os
import numpy as np
import shutil

import os
import shutil
from tqdm import tqdm

def save_transforms(transform_path, output_dir, image_path, i, j, k):
    """
    指定された変形パラメータファイルを、改名して出力ディレクトリにコピーする。

    Args:
    - transform_path: 変形パラメータファイルへのパス
    - output_dir: 変形パラメータファイルを保存するディレクトリ
    - image_path: 入力画像へのパス（ファイル名のプレフィックスに使用）
    - i: 反復回数
    - j: 変形の種類を示すインデックス
    - k: 画像のインデックス
    """
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 入力画像のファイル名を取得（拡張子なし）
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 新しいファイル名を構築
    transform_type = "Affine" if transform_path.endswith("Affine.mat") else "Warp"
    new_file_name = f"{img_name}_iter{i}_type{j}_{transform_type}_img{k}.mat"
    
    # 出力ディレクトリへの完全なパスを構築
    output_path = os.path.join(output_dir, new_file_name)
    
    # ファイルをコピー
    shutil.copy(transform_path, output_path)


def build_template_and_save_transforms(
    initial_template=None,
    image_list=None,
    image_path_list=None,
    iterations=3,
    gradient_step=0.2,
    blending_weight=0.75,
    weights=None,
    useNoRigid=True,
    output_transforms_dir=None,  # 変形マップを保存するディレクトリのパス
    **kwargs
):
    if "type_of_transform" not in kwargs:
        type_of_transform = "SyN"
    else:
        type_of_transform = kwargs.pop("type_of_transform")

    if weights is None:
        weights = np.repeat(1.0 / len(image_list), len(image_list))
    weights = [x / sum(weights) for x in weights]
    if initial_template is None:
        initial_template = image_list[0] * 0
        for i in range(len(image_list)):
            temp = image_list[i] * weights[i]
            temp = resample_image_to_target(temp, initial_template)
            initial_template = initial_template + temp

    xavg = initial_template.clone()
    for i in trange(iterations):
        affinelist = []
        for k, img in enumerate(image_list):
            w1 = registration(
                xavg, img, type_of_transform=type_of_transform, **kwargs
            )
            L = len(w1["fwdtransforms"])
            affinelist.append(w1["fwdtransforms"][L-1])

            # ここで変形マップを保存
            if output_transforms_dir is not None:
                for j, transform_path in enumerate(w1["fwdtransforms"]):
                    if ".mat" in transform_path:  # AffineまたはWarp変換ファイルのみを対象
                        save_transforms(transform_path, output_transforms_dir, image_path_list[k], i, j, k)


            if k == 0:
                if L == 2:
                    wavg = ants.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = w1["warpedmovout"] * weights[k]
            else:
                if L == 2:
                    wavg = wavg + ants.image_read(w1["fwdtransforms"][0]) * weights[k]
                xavgNew = xavgNew + w1["warpedmovout"] * weights[k]

        if useNoRigid:
            avgaffine = utils.average_affine_transform_no_rigid(affinelist)
        else:
            avgaffine = utils.average_affine_transform(affinelist)
        afffn = mktemp(suffix=".mat")
        tio.write_transform(avgaffine, afffn)

        if L == 2:
            print(wavg.abs().mean())
            wscl = (-1.0) * gradient_step
            wavg = wavg * wscl
            # apply affine to the nonlinear?
            # need to save the average
            wavgA = apply_transforms(fixed = xavgNew, moving = wavg, imagetype=1, transformlist=afffn, whichtoinvert=[1])
            wavgfn = mktemp(suffix=".nii.gz")
            iio.image_write(wavgA, wavgfn)
            xavg = apply_transforms(fixed=xavgNew, moving=xavgNew, transformlist=[wavgfn, afffn], whichtoinvert=[0, 1])
        else:
            xavg = apply_transforms(fixed=xavgNew, moving=xavgNew, transformlist=[afffn], whichtoinvert=[1])
            
        os.remove(afffn)
        if blending_weight is not None:
            xavg = xavg * blending_weight + utils.iMath(xavg, "Sharpen") * (
                1.0 - blending_weight
            )

    return xavg

def apply_transform_to_images(image_names, transforms_dir, reference_image):
    """
    Args:
    - image_names: List of image file paths
    - transforms_dir: Directory containing the transformation files
    - reference_image: Average image to which the transformations are applied
    """
    transformed_images = []

    for img_name in tqdm(image_names):
        base_name = os.path.splitext(os.path.basename(img_name))[0]

        # 対応する変形ファイルを検索（反復回数iが最大のものを選択）
        transform_files = glob.glob(os.path.join(transforms_dir, f"{base_name}_iter*"))
        if not transform_files:
            raise FileNotFoundError(f"No transform files found for {img_name}")

        # 反復回数を取得し、最大のものを選択
        max_iter_transform = max(transform_files, key=lambda x: int(x.split('_iter')[1].split('_')[0]))

        # 変形を適用
        img = ants.image_read(img_name)
        transformed_img = ants.apply_transforms(fixed=reference_image, moving=img, transformlist=[max_iter_transform], interpolator="nearestNeighbor")
        transformed_images.append(transformed_img)

    return transformed_images



def create_majority_vote_label_image(transformed_images, transformed_images_npy):
    """
    変形後の画像リストから多数決ラベル画像を生成する。
    """
    # 画素値を整数型にキャスト
    image_stack = transformed_images_npy.astype(np.uint16)
    
    # 各画素において最も頻度が高いラベル値を計算
    majority_vote_labels = np.apply_along_axis(lambda x: np.bincount(x, minlength=256).argmax(), axis=-1, arr=image_stack)
    majority_vote_labels = majority_vote_labels.astype(np.uint8 if np.max(majority_vote_labels) <= 255 else np.uint16)

    return majority_vote_labels


def save_transformed_images_as_4d_nifti(transformed_images, output_file_path, affine=np.eye(4)):

    """
    変形後の画像リストを4D NIfTIファイルとして保存する。

    Args:
    - transformed_images: 変形後のANTsImageオブジェクトのリスト
    - output_file_path: 保存する4D NIfTIファイルのパス
    """
    # 画像リストから3D NumPy配列を抽出し、4D配列を作成
    image_data = np.stack([img.numpy() for img in transformed_images], axis=-1)
    
    # 4D NumPy配列をNIfTIイメージとして保存
    nifti_image = nib.Nifti1Image(image_data, affine=affine)
    nib.save(nifti_image, output_file_path)
    return image_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True, help="Directory with the input NIfTI files.")
    parser.add_argument("-o", "--output", default=None, help="Output dir path to save the mean organ model.")
    parser.add_argument("-s", "--string", default="gth", help="String that should be included in input file names.")
    args = parser.parse_args()

    # Set the output directory
    if args.output is None:
        args.output = os.path.normpath(args.input_dir) + "_mean-shape"
        print(f"Output directory not specified. Using {args.output} as the output directory.")
    os.makedirs(args.output, exist_ok=True)

    # Get a list of all label files in the input directory
    label_files = [f for f in glob.glob(os.path.join(args.input_dir, "*")) if f.find(args.string) >= 0 and f.endswith(".nii.gz")]
    label_images = [ants.image_read(path) for path in label_files]
    # Perform groupwise registration
    
    output_transforms_dir = os.path.join(args.output, "transforms")
    template_image = build_template_and_save_transforms(image_list = label_images, image_path_list=label_files, iterations=3, output_transforms_dir=output_transforms_dir)
    ants.image_write(template_image, os.path.join(args.output, "template.nii.gz"))
    print(f"Template image saved to {os.path.join(args.output, 'template.nii.gz')}.")

    # 変形マップのパスをリスト化
    # transform_paths = [os.path.join(output_transforms_dir, f"transform_{i}_0GenericAffine.mat") for i in range(len(label_files))]

    # 変形を適用
    print("Applying transforms to images...")
    transformed_images = apply_transform_to_images(label_files, output_transforms_dir, template_image)
    transformed_images_npy = save_transformed_images_as_4d_nifti(transformed_images, os.path.join(args.output, "transformed_images_as_4d.nii.gz"))

    majority_image = create_majority_vote_label_image(transformed_images, transformed_images_npy)
    nib.save(nib.Nifti1Image(majority_image, affine=np.eye(4)), os.path.join(args.output, "majority_vote_labels.nii.gz"))