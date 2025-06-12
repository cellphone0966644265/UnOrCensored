
import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """
    Đọc và cài đặt các gói từ file requirements.txt.
    """
    req_path = Path(__file__).parent / "requirements.txt"
    if not req_path.exists():
        print(f"LỖI: không tìm thấy file requirements.txt tại {req_path}")
        return

    print("--- Bắt đầu cài đặt các thư viện cần thiết ---")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
        print("--- Cài đặt thư viện thành công ---")
    except subprocess.CalledProcessError as e:
        print(f"LỖI: Không thể cài đặt thư viện. Lỗi: {e}")
        sys.exit(1)

def download_models():
    """
    Sử dụng gdown để tải các mô hình pre-trained từ Google Drive.
    """
    print("\n--- Bắt đầu tải các mô hình pre-trained ---")
    # [span_0](start_span)ID của thư mục Google Drive chứa các mô hình[span_0](end_span)
    gdrive_folder_id = "1YoyNPQjN_DgfO7oDpWc0sFUL8oP54HMO"

    # [span_1](start_span)[span_2](start_span)Thư mục để chứa các mô hình tải về[span_1](end_span)[span_2](end_span)
    output_dir = Path(__file__).parent / "pre_trained_models"
    output_dir.mkdir(exist_ok=True)

    command = ["gdown", "--folder", gdrive_folder_id, "-O", str(output_dir), "--continue"]

    print(f"Đang thực thi lệnh: {' '.join(command)}")

    try:
        # gdown sẽ tải toàn bộ nội dung của thư mục vào output_dir
        subprocess.run(command, check=True)
        print(f"--- Tải mô hình thành công vào thư mục {output_dir} ---")

        # Xác nhận các file đã được tải về như trong tài liệu
        verify_downloaded_files(output_dir)

    except FileNotFoundError:
        print("LỖI: Lệnh 'gdown' không tồn tại.")
        print("Vui lòng cài đặt gdown bằng lệnh: pip install gdown")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"LỖI: Quá trình tải mô hình thất bại. Lỗi: {e}")
        sys.exit(1)

def verify_downloaded_files(model_dir):
    """
    [span_3](start_span)Kiểm tra sự tồn tại của các file model và yaml cần thiết.[span_3](end_span)
    """
    print("\n--- Bắt đầu xác nhận các file mô hình ---")
    required_files = [
        'add_youknow.pth', 'add_youknow_structure.yaml',
        'clean_youknow_video.pth', 'clean_youknow_video_structure.yaml',
        'clean_youknow_image.pth', 'clean_youknow_image_structure.yaml',
        'mosaic_position.pth', 'mosaic_position_structure.yaml'
    ]

    missing_files = []
    for f in required_files:
        if not (model_dir / f).exists():
            missing_files.append(f)

    if not missing_files:
        print("--- Xác nhận thành công! Tất cả các file cần thiết đều có mặt. ---")
    else:
        print("CẢNH BÁO: Thiếu các file sau đây trong thư mục pre_trained_models/:")
        for f in missing_files:
            print(f" - {f}")
        print("Script có thể không hoạt động đúng. Vui lòng kiểm tra lại quá trình tải về.")

if __name__ == "__main__":
    install_requirements()
    download_models()

    print("\n--- Cài đặt và chuẩn bị dự án hoàn tất! ---")

