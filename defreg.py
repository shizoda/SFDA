import ants, pdb, time, os
import numpy as np
import argparse
from tqdm import tqdm

def ants_to_numpy(image):
    # 画像データをNumPy配列として抽出
    image_data = image.numpy()
    
    # ヘッダ情報の抽出
    header_info = {
        "spacing": image.spacing,  # ピクセル間距離
        "origin": image.origin,  # 画像の原点
        "direction": image.direction,  # 画像の向き
        "affine": image.get_affine()  # アフィン行列
    }
    return image_data, header_info

def numpy_to_ants(image_data, spacing=[1, 1, 1], origin=[0, 0, 0], direction=np.diag([1, 1, 1, 1]), affine=np.diag([1, 1, 1, 1])):
    # NumPy配列から画像データを作成
    image = ants.from_numpy(image_data)
    
    # ヘッダ情報を設定
    image.set_spacing(spacing)
    image.set_origin(origin)
    image.set_direction(direction)
    image.set_affine(affine)
    return image

def register_multiclass_labels_with_single_deformation(fixed_image, moving_image, type_of_transform="SyN",
                                                       grad_step=0.2, flow_sigma=3, total_sigma=0,
                                                       syn_metric='mattes', syn_sampling=32,
                                                       reg_iterations=(40, 20, 0),
                                                       verbose=0, return_numpy=False, **kwargs):

    unique_labels = np.unique(moving_image.numpy())[1:]  # 最初の要素0をスキップ

    # 変形場の計算
    registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform=type_of_transform,
                                     grad_step=grad_step, flow_sigma=flow_sigma, total_sigma=total_sigma,
                                     syn_metric=syn_metric, syn_sampling=syn_sampling,
                                     reg_iterations=reg_iterations, verbose=verbose > 0, **kwargs)

    # 変形の適用（最近傍法でラベルを変形後の空間にマッピング）
    output_labels = ants.apply_transforms(fixed=fixed_image, moving=moving_image,
                                                     transformlist=registration['fwdtransforms'],
                                                     interpolator='nearestNeighbor')

    # データタイプの選択
    if np.max(unique_labels) <= 255:
        output_labels = output_labels.numpy().astype(np.uint8)
    else:
        output_labels = output_labels.numpy().astype(np.uint16)
    
    # アフィン行列を固定画像と同じものを使用して保存
    return  output_labels if return_numpy else ants.from_numpy(output_labels, spacing=fixed_image.spacing, origin=fixed_image.origin, direction=fixed_image.direction)


def main():
    parser = argparse.ArgumentParser(description="Register two images using ANTsPy.")
    parser.add_argument("-f", "--fixed_image", required=True, help="Path to the fixed image.")
    parser.add_argument("-m", "--moving_image", required=True, help="Path to the moving image.")
    parser.add_argument("-o", "--output_image", default=None, help="Path to save the transformed moving image.")
    parser.add_argument("-t", "--transformation_type", default='Elastic',
                        help="Type of transformation. SyN is another recommendation.")
    args = parser.parse_args()

    if args.output_image is None:
        args.output_image = os.path.join(os.path.dirname(args.fixed_image),  "reg_" + args.transformation_type, os.path.basename(args.fixed_image).replace('.nii.gz', f'_reg_{args.transformation_type}.nii.gz'))

    os.makedirs(os.path.dirname(args.output_image), exist_ok=True)

    fixed_image = ants.image_read(args.fixed_image)
    moving_image = ants.image_read(args.moving_image)

    print(f"Registering {os.path.basename(args.moving_image)} to {os.path.basename(args.fixed_image)} using {args.transformation_type} transformation.")
    output_labels = register_multiclass_labels_with_single_deformation(fixed_image, moving_image, args.transformation_type, aff_metric="meansquares", syn_metric="meansquares")
    ants.image_write(output_labels, args.output_image)

if __name__ == "__main__":
    main()
