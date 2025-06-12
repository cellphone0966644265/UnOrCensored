
import argparse, sys, time, shutil, glob
from pathlib import Path
from collections import OrderedDict

import torch, numpy as np, cv2

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts_AI import ffmpeg_util, image_processing_util as ipu, models as custom_models

def _fix_bisenet_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'context_path.features' in k: name = k.replace('context_path.features.', 'context_path.')
        else: name = k
        new_state_dict[name] = v
    return new_state_dict

def _prepare_temp_dirs(project_root, for_image=False):
    """Chuẩn bị và dọn dẹp các thư mục tạm theo đúng thiết kế."""
    tmp_dir = project_root / "tmp"
    img_proc_dir = tmp_dir / "image_processing"
    vid_proc_dir = tmp_dir / "video_processing"

    if for_image:
        if img_proc_dir.exists(): shutil.rmtree(img_proc_dir)
        img_proc_dir.mkdir(parents=True, exist_ok=True)
        (img_proc_dir / "input").mkdir()
        (img_proc_dir / "mask").mkdir()
        return img_proc_dir
    else:
        if vid_proc_dir.exists(): shutil.rmtree(vid_proc_dir)
        vid_proc_dir.mkdir(parents=True, exist_ok=True)
        (vid_proc_dir / "audio").mkdir()
        (vid_proc_dir / "input_chunks").mkdir()
        (vid_proc_dir / "input_frames").mkdir()
        (vid_proc_dir / "masks").mkdir()
        (vid_proc_dir / "output_frames").mkdir()
        (vid_proc_dir / "output_processing").mkdir()
        return vid_proc_dir

def process_image(args, project_root):
    print("--- Bắt đầu tác vụ ADD cho ảnh ---")
    _prepare_temp_dirs(project_root, for_image=True)

    model_dir = project_root / "pre_trained_models"
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id != '-1' and torch.cuda.is_available() else "cpu")

    print("Đang tải model...")
    add_net = custom_models.BiSeNet(num_classes=1, context_path='resnet18', train_flag=False).to(device)
    state_dict_add = torch.load(model_dir / 'add_youknow.pth', map_location='cpu')
    add_net.load_state_dict(_fix_bisenet_state_dict(state_dict_add), strict=False)
    add_net.eval()

    img = ipu.imread(str(args.input_path))
    mask = ipu.run_segment(img, add_net, gpu_id=args.gpu_id)
    mask = ipu.mask_threshold(mask, 128, ex_mun=args.mask_extend)

    print("Thêm mosaic...")
    img_result = ipu.addmosaic_base(img, mask, n=args.mosaic_size, mod=args.mosaic_mod)

    output_dir = Path(args.output_path or (project_root / "output")); output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f"{Path(args.input_path).stem}_added{Path(args.input_path).suffix}"
    ipu.imwrite(str(output_path), img_result)
    print(f"Lưu ảnh thành công tại: {output_path}")

def process_video(args, project_root):
    print("--- Bắt đầu tác vụ ADD cho video ---")
    tmp_dir = _prepare_temp_dirs(project_root, for_image=False)
    frames_in_dir = tmp_dir / "input_frames"
    frames_out_dir = tmp_dir / "output_frames"

    try:
        model_dir = project_root / "pre_trained_models"
        device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id != '-1' and torch.cuda.is_available() else "cpu")
        add_net = custom_models.BiSeNet(num_classes=1, context_path='resnet18', train_flag=False).to(device)
        state_dict_add = torch.load(model_dir / 'add_youknow.pth', map_location='cpu')
        add_net.load_state_dict(_fix_bisenet_state_dict(state_dict_add), strict=False)
        add_net.eval()

        info = ffmpeg_util.get_video_info(args.input_path)
        audio_path = tmp_dir / "audio.aac"
        ffmpeg_util.extract_audio(args.input_path, str(audio_path), args.start_time, args.end_time)
        ffmpeg_util.extract_frames(args.input_path, str(frames_in_dir / "frame_%06d.png"), start_time=args.start_time, end_time=args.end_time)
        frames = sorted(list(frames_in_dir.glob("*.png")))
        if not frames: raise RuntimeError("Không trích xuất được khung hình nào từ video.")

        for idx, frame_path in enumerate(frames):
            print(f"\rĐang xử lý frame {idx+1}/{len(frames)}...", end="")
            img = ipu.imread(str(frame_path))
            mask = ipu.run_segment(img, add_net, gpu_id=args.gpu_id)
            mask = ipu.mask_threshold(mask, 128, ex_mun=args.mask_extend)
            img_result = ipu.addmosaic_base(img, mask, n=args.mosaic_size, mod=args.mosaic_mod)
            ipu.imwrite(str(frames_out_dir / frame_path.name), img_result)

        output_dir = Path(args.output_path or (project_root / "output")); output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"{Path(args.input_path).stem}_added{Path(args.input_path).suffix}"
        ffmpeg_util.build_video_from_frames(str(frames_out_dir / "frame_%06d.png"), str(audio_path) if audio_path.exists() else None, str(output_path), info['fps'], info['resolution'])
        print(f"\nLưu video thành công tại: {output_path}")
    finally:
        if tmp_dir.exists(): shutil.rmtree(tmp_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--start_time', type=str)
    parser.add_argument('--end_time', type=str)
    parser.add_argument('--mosaic_mod', type=str, default='squa_avg')
    parser.add_argument('--mosaic_size', type=int, default=0)
    parser.add_argument('--mask_extend', type=int, default=10)
    args, _ = parser.parse_known_args()

    # *** SỬA LỖI: Xử lý logic end_time = 00:00:00 ***
    if args.end_time == "00:00:00":
        args.end_time = None

    file_ext = Path(args.input_path).suffix.lower()
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        process_image(args, project_root)
    elif file_ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
        process_video(args, project_root)
    else:
        print(f"Định dạng file không được hỗ trợ: {file_ext}"); sys.exit(1)

if __name__ == '__main__':
    main()
