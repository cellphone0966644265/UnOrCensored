
import argparse, sys, shutil, random
from pathlib import Path
from tqdm import tqdm
import cv2

# Thêm đường dẫn gốc và import
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts_AI import image_processing_util as ipu

def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị bộ dữ liệu (ảnh đã làm mờ) cho các model 'clean'.")
    parser.add_argument('--data_path', required=True, type=str, help="Đường dẫn đến thư mục 'data' chứa 'origin_image' và 'mask'.")
    parser.add_argument('--out_size', type=int, default=256, help="Kích thước ảnh đầu ra.")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    origin_dir = data_path / "origin_image"
    mask_dir = data_path / "mask"
    mosaiced_dir = data_path / "mosaiced" # Thư mục đầu ra theo thiết kế

    if not origin_dir.is_dir() or not mask_dir.is_dir():
        print(f"LỖI: Không tìm thấy thư mục 'origin_image' hoặc 'mask' bên trong '{data_path}'")
        sys.exit(1)

    if mosaiced_dir.exists():
        print(f"Thư mục '{mosaiced_dir}' đã tồn tại. Xóa đi để tạo mới...")
        shutil.rmtree(mosaiced_dir)
    mosaiced_dir.mkdir(exist_ok=True)

    print(f"Bắt đầu tạo bộ dữ liệu 'mosaiced'...")
    image_paths = sorted(list(origin_dir.glob('*.*')))

    for img_path in tqdm(image_paths, desc="Đang xử lý các ảnh"):
        mask_path = mask_dir / f"{img_path.stem}.png"
        if not mask_path.exists(): mask_path = mask_dir / f"{img_path.stem}.jpg"
        if not mask_path.exists(): continue

        img_origin = ipu.imread(str(img_path))
        mask = ipu.imread(str(mask_path), mod='gray')

        mosaic_size = random.randint(8, 20)
        img_mosaic = ipu.addmosaic_base(img_origin, mask, n=mosaic_size, mod='squa_avg')

        img_mosaic_resized = cv2.resize(img_mosaic, (args.out_size, args.out_size))

        ipu.imwrite(str(mosaiced_dir / img_path.name), img_mosaic_resized)

    print(f"✅ Hoàn tất! Đã tạo bộ dữ liệu 'mosaiced' tại: {mosaiced_dir}")

if __name__ == '__main__':
    main()
