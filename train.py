
import argparse
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="UnOrCensored - Bộ điều phối huấn luyện thông minh")

    # Parser chính chỉ cần các tham số cốt lõi để điều phối
    parser.add_argument('--model_name', required=True, choices=['mosaic_position', 'add_youknow', 'clean_youknow_image', 'clean_youknow_video'])
    parser.add_argument('--data_path', required=True, type=str, help="Đường dẫn đến thư mục 'data' của dự án.")

    # Thu thập tất cả các tham số khác để truyền cho script con
    args, other_args = parser.parse_known_args()

    # Xác định đường dẫn gốc của dự án từ vị trí của file train.py
    project_root = Path(__file__).resolve().parent
    data_path = Path(args.data_path)

    # --- BƯỚC 1: TỰ ĐỘNG CHUẨN BỊ DỮ LIỆU NẾU CẦN ---
    if args.model_name in ['clean_youknow_image', 'clean_youknow_video']:
        # Đối với các model 'clean', chúng ta cần có dữ liệu đã được làm mờ
        # Script prepare_dataset.py sẽ tạo ra thư mục 'mosaiced' bên trong data_path
        mosaiced_dir = data_path / "mosaiced"

        # Chỉ chạy prepare nếu thư mục chưa tồn tại hoặc rỗng
        if not mosaiced_dir.exists() or not any(mosaiced_dir.iterdir()):
            print("--- Phát hiện model 'clean' và thiếu dữ liệu đã xử lý. ---")
            print("--- Tự động gọi module prepare_dataset.py ---")

            prepare_command = [
                sys.executable,
                str(project_root / "scripts_AI" / "prepare_dataset.py"),
                "--data_path", str(data_path)
            ]

            try:
                subprocess.run(prepare_command, check=True)
                print("--- Hoàn tất chuẩn bị dữ liệu. ---")
            except subprocess.CalledProcessError:
                print("LỖI: Quá trình chuẩn bị dữ liệu thất bại. Dừng lại.")
                sys.exit(1)
        else:
            print(f"ℹ️ Đã tìm thấy dữ liệu tại '{mosaiced_dir}'. Sẵn sàng để huấn luyện.")

    # --- BƯỚC 2: GỌI SCRIPT HUẤN LUYỆN CHUYÊN BIỆT ---
    script_to_run = project_root / "scripts_AI" / f"train_{args.model_name}.py"

    if not script_to_run.exists():
        print(f"LỖI: Script huấn luyện '{script_to_run}' không tồn tại.")
        sys.exit(1)

    # Xây dựng lệnh cuối cùng, truyền tất cả tham số ban đầu
    # other_args đã chứa tất cả các cờ và giá trị chưa được parse, ví dụ: [--folder_path, ..., --lr, ...]
    final_command = [sys.executable, str(script_to_run)] + other_args

    # Đảm bảo các tham số cốt lõi luôn được truyền đúng
    if '--data_path' not in str(other_args):
        final_command.extend(['--data_path', args.data_path])
    if '--model_name' not in str(other_args):
         final_command.extend(['--model_name', args.model_name])

    print(f"\n--- Điều phối đến tác vụ huấn luyện model: '{args.model_name}' ---")
    print(f"Lệnh thực thi: {' '.join(final_command)}")

    try:
        # Chạy subprocess
        subprocess.run(final_command, check=True)
        print(f"--- Tác vụ huấn luyện '{args.model_name}' đã hoàn thành! ---")
    except subprocess.CalledProcessError:
        print(f"--- Tác vụ huấn luyện '{args.model_name}' đã thất bại. ---")
        sys.exit(1)

if __name__ == '__main__':
    main()
