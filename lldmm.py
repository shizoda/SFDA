import ants, pdb, time
import numpy as np
import argparse

def register_multiclass_labels(fixed_image_path, moving_image_path, output_image_path, transformation_type='SyN'):
    # 画像の読み込み
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)
    
    
    # 移動画像のユニークなクラスラベルを取得（背景0を除く）
    unique_labels = np.unique(moving_image.numpy())[1:]  # 最初の要素0をスキップ

    # 出力用の空の配列を準備
    output_labels = np.zeros_like(moving_image.numpy(), dtype=np.uint16)

    # 各クラスラベルに対して変形登録を実行
    for label in unique_labels:
        mask_fixed = fixed_image == label
        mask_moving = moving_image == label
        mask_moving_adjusted = ants.from_numpy(mask_moving.numpy(), spacing=fixed_image.spacing, origin = fixed_image.origin, direction=fixed_image.direction)
        
        # どちらかにしか存在しないクラスの確認
        if np.sum(mask_fixed.numpy()) == 0 or np.sum(mask_moving.numpy()) == 0:
            print(f"Error: Label {label} exists only in one image. Skipping...")
            continue
        
        # 変形登録の実行
        print(f"Processing label: {label}")
        start = time.time()
        registration = ants.registration(fixed=mask_fixed, moving=mask_moving_adjusted,
                                         type_of_transform=transformation_type)
        transformed_mask = registration['warpedmovout'].numpy()
        print("Registration completed.", time.time() - start, "sec")
        
        # ラベル番号に応じて結果を統合
        output_labels[transformed_mask > 0.5] = label  # 変形後のマスクを適用

    # データタイプの選択
    if np.max(unique_labels) <= 255:
        output_labels = output_labels.astype(np.uint8)
    else:
        output_labels = output_labels.astype(np.uint16)
    
    # アフィン行列を固定画像と同じものを使用して保存
    ants.image_write(ants.from_numpy(output_labels, spacing=fixed_image.spacing, origin=fixed_image.origin, direction=fixed_image.direction), output_image_path)

def main():
    parser = argparse.ArgumentParser(description="Register two images using ANTsPy.")
    parser.add_argument("-f", "--fixed_image", required=True, help="Path to the fixed image.")
    parser.add_argument("-m", "--moving_image", required=True, help="Path to the moving image.")
    parser.add_argument("-o", "--output_image", required=True, help="Path to save the transformed moving image.")
    parser.add_argument("-t", "--transformation_type", default='SyN', choices=['SyN', 'Affine', 'Rigid'],
                        help="Type of transformation. Default is 'SyN'.")
    args = parser.parse_args()

    # 画像登録関数の呼び出し
    register_multiclass_labels(args.fixed_image, args.moving_image, args.output_image, args.transformation_type)

if __name__ == "__main__":
    main()
