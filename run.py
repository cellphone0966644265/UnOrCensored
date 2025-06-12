
import argparse
import os
import sys
import subprocess
from pathlib import Path

def main():
    """
    Script điều phối chính với đầy đủ các tham số tinh chỉnh.
    """
    parser = argparse.ArgumentParser(
        description="UnOrCensored - Hệ thống xử lý ảnh và video tiên tiến.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Tham số chung ---
    general_group = parser.add_argument_group('Tham số chung')
    general_group.add_argument('--task_name', type=str, required=True, choices=['clean_youknow', 'add_youknow'],
                               help='Tên tác vụ cần thực hiện.')
    general_group.add_argument('--input_path', type=str, required=True,
                               help='Đường dẫn đến file ảnh hoặc video đầu vào.')
    general_group.add_argument('--output_path', type=str, default=None,
                               help='(Tùy chọn) Đường dẫn thư mục để lưu file kết quả. Mặc định là ./output/.')
    general_group.add_argument('--gpu_id', type=str, default='0', help='ID của GPU để sử dụng (vd: 0), -1 cho CPU.')

    # --- Tham số Video ---
    video_group = parser.add_argument_group('Tham số cho Video')
    video_group.add_argument('--start_time', type=str, default=None,
                             help='(Tùy chọn) Thời gian bắt đầu xử lý (vd: 00:01:23).')
    video_group.add_argument('--end_time', type=str, default=None,
                             help='(Tùy chọn) Thời gian kết thúc xử lý (vd: 00:02:50).')
    video_group.add_argument('--medfilt_num', type=int, default=11,
                             help='(Chỉ cho video) Kích thước cửa sổ lọc trung vị để làm mượt chuyển động. Phải là số lẻ.')

    # --- Tham số cho tác vụ "ADD" ---
    add_group = parser.add_argument_group('Tham số cho tác vụ "add_youknow"')
    add_group.add_argument('--mosaic_mod', type=str, default='squa_avg',
                           choices=['squa_avg', 'squa_random', 'rect_avg', 'random'],
                           help='Kiểu mosaic để thêm vào.')
    add_group.add_argument('--mosaic_size', type=int, default=0,
                           help='Kích thước của mỗi ô mosaic. 0 = tự động.')
    add_group.add_argument('--mask_extend', type=int, default=10,
                           help='Số pixel mở rộng vùng mặt nạ ra xung quanh.')

    # --- Tham số cho tác vụ "CLEAN" ---
    clean_group = parser.add_argument_group('Tham số cho tác vụ "clean_youknow"')
    clean_group.add_argument('--mask_threshold', type=int, default=64,
                             help='Ngưỡng nhạy cảm để phát hiện vùng mosaic (0-255). Càng nhỏ càng nhạy.')
    clean_group.add_argument('--ex_mult', type=float, default=1.1,
                             help='Hệ số mở rộng vùng ảnh crop đưa vào model làm sạch.')
    clean_group.add_argument('--no_feather', action='store_true',
                             help='Tắt tính năng làm mịn và hòa trộn biên, giúp chạy nhanh hơn.')
    clean_group.add_argument('--all_mosaic_area', action='store_true',
                             help='Tìm và xử lý tất cả các vùng mosaic thay vì chỉ vùng lớn nhất.')

    args, unknown = parser.parse_known_args()

    # 1. Kiểm tra sự tồn tại của file đầu vào
    if not os.path.exists(args.input_path):
        print(f"LỖI: File đầu vào không tồn tại tại '{args.input_path}'")
        sys.exit(1)

    # 2. Xác định và gọi script xử lý chuyên biệt
    script_to_run = Path("scripts_AI") / f"run_{args.task_name}.py"

    if not script_to_run.exists():
        print(f"LỖI: Script xử lý '{script_to_run}' không tồn tại.")
        sys.exit(1)

    # 3. Xây dựng lệnh cho script con, truyền tất cả các tham số
    command = [sys.executable, str(script_to_run)]
    for arg, value in vars(args).items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    command.append(f'--{arg}')
            else:
                command.extend([f'--{arg}', str(value)])

    print(f"--- Điều phối đến tác vụ '{args.task_name}' ---")
    print(f"Lệnh thực thi: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
        print(f"--- Tác vụ '{args.task_name}' đã hoàn thành thành công! ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Tác vụ '{args.task_name}' đã thất bại. ---")
        sys.exit(1)

if __name__ == '__main__':
    main()
