
import argparse, sys, time, shutil, glob
from pathlib import Path
from collections import OrderedDict
from tqdm import tqdm

import torch, torch.nn as nn, numpy as np, cv2

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts_AI import ffmpeg_util, image_processing_util as ipu, models as custom_models, filt_util

def _fix_bvdnet_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'parametrizations.weight.original' in k:
            name = k.replace('parametrizations.weight.original', 'weight_orig')
        elif 'parametrizations.weight.0._u' in k:
            name = k.replace('parametrizations.weight.0._u', 'weight_u')
        elif 'parametrizations.weight.0._v' in k:
            name = k.replace('parametrizations.weight.0._v', 'weight_v')
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def _prepare_temp_dirs(project_root, for_image=False):
    tmp_dir = project_root / "tmp"
    img_proc_dir = tmp_dir / "image_processing"
    vid_proc_dir = tmp_dir / "video_processing"
    if for_image:
        if img_proc_dir.exists(): shutil.rmtree(img_proc_dir)
        img_proc_dir.mkdir(parents=True, exist_ok=True)
        (img_proc_dir / "input").mkdir(); (img_proc_dir / "mask").mkdir()
        return img_proc_dir
    else:
        if vid_proc_dir.exists(): shutil.rmtree(vid_proc_dir)
        vid_proc_dir.mkdir(parents=True, exist_ok=True)
        (vid_proc_dir / "audio").mkdir(); (vid_proc_dir / "input_frames").mkdir()
        (vid_proc_dir / "masks").mkdir(); (vid_proc_dir / "output_frames").mkdir()
        return vid_proc_dir

def process_image(args, project_root):
    # Phần xử lý ảnh giữ nguyên, không thay đổi
    print("--- Bắt đầu tác vụ CLEAN cho ảnh ---")
    tmp_dir = _prepare_temp_dirs(project_root, for_image=True)
    try:
        model_dir = project_root / "pre_trained_models"
        device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id != '-1' and torch.cuda.is_available() else "cpu")
        mask_gen_net = custom_models.BiSeNet(num_classes=1, context_path='resnet18').to(device)
        clean_net = custom_models.ResnetGenerator(input_nc=3, output_nc=3, n_blocks=9, use_dropout=True).to(device)
        state_dict_mask = torch.load(model_dir / 'mosaic_position.pth', map_location='cpu')
        new_state_dict_mask = OrderedDict()
        for k, v in state_dict_mask.items():
            name = k.replace('context_path.features.', 'context_path.') if 'context_path.features' in k else k
            new_state_dict_mask[name] = v
        mask_gen_net.load_state_dict(new_state_dict_mask, strict=False)
        clean_net.load_state_dict(torch.load(model_dir / 'clean_youknow_image.pth', map_location='cpu'))
        mask_gen_net.eval(); clean_net.eval()
        img = ipu.imread(str(args.input_path))
        mask = ipu.run_segment(img, mask_gen_net, gpu_id=args.gpu_id)
        mask = ipu.mask_threshold(mask, args.mask_threshold, ex_mun=10)
        if not args.all_mosaic_area: mask = ipu.find_mostlikely_ROI(mask)
        x, y, size, _ = ipu.boundingSquare(mask, Ex_mul=args.ex_mult)
        if size > 0:
            y_start, y_end = max(0, y - size), min(img.shape[0], y + size)
            x_start, x_end = max(0, x - size), min(img.shape[1], x + size)
            img_crop = img[y_start:y_end, x_start:x_end]
            if img_crop.size > 0:
                img_fake = ipu.run_pix2pix_generator(img_crop, clean_net, gpu_id=args.gpu_id)
                img_result = ipu.replace_mosaic(img, img_fake, mask, (x_start, y_start, x_end, y_end), no_feather=args.no_feather)
            else: img_result = img
        else: img_result = img
        output_dir = Path(args.output_path or (project_root / "output")); output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"{Path(args.input_path).stem}_cleaned{Path(args.input_path).suffix}"
        ipu.imwrite(str(output_path), img_result)
        print(f"Lưu ảnh thành công tại: {output_path}")
    finally:
        if tmp_dir.exists(): shutil.rmtree(tmp_dir)

def process_video(args, project_root):
    print("--- Bắt đầu tác vụ CLEAN cho video ---")
    tmp_dir = _prepare_temp_dirs(project_root, for_image=False)
    frames_in_dir = tmp_dir / "input_frames"; frames_out_dir = tmp_dir / "output_frames"; masks_dir = tmp_dir / "masks"
    try:
        model_dir = project_root / "pre_trained_models"
        device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id != '-1' and torch.cuda.is_available() else "cpu")
        mask_gen_net = custom_models.BiSeNet(num_classes=1, context_path='resnet18').to(device)
        clean_net = custom_models.BVDNet(n_blocks=4).to(device)
        state_dict_mask = torch.load(model_dir / 'mosaic_position.pth', map_location='cpu')
        new_state_dict_mask = OrderedDict()
        for k, v in state_dict_mask.items():
            name = k.replace('context_path.features.', 'context_path.') if 'context_path.features' in k else k
            new_state_dict_mask[name] = v
        mask_gen_net.load_state_dict(new_state_dict_mask, strict=False)
        state_dict_video = torch.load(model_dir / 'clean_youknow_video.pth', map_location='cpu')
        clean_net.load_state_dict(_fix_bvdnet_state_dict(state_dict_video), strict=False)
        mask_gen_net.eval(); clean_net.eval()
        
        info = ffmpeg_util.get_video_info(args.input_path)
        audio_path = tmp_dir / "audio" / "audio.aac"
        ffmpeg_util.extract_audio(args.input_path, str(audio_path), args.start_time, args.end_time)
        ffmpeg_util.extract_frames(args.input_path, str(frames_in_dir / "frame_%06d.png"), start_time=args.start_time, end_time=args.end_time)
        frames = sorted(list(frames_in_dir.glob("*.png")))
        if not frames: raise RuntimeError("Không trích xuất được khung hình nào từ video.")
        
        positions = []
        for frame_path in tqdm(frames, desc="Tạo mặt nạ"):
            img = ipu.imread(str(frame_path)); mask = ipu.run_segment(img, mask_gen_net, gpu_id=args.gpu_id)
            mask = ipu.mask_threshold(mask, args.mask_threshold, 10)
            if not args.all_mosaic_area: mask = ipu.find_mostlikely_ROI(mask)
            ipu.imwrite(str(masks_dir / frame_path.name), mask)
            x, y, size, _ = ipu.boundingSquare(mask, args.ex_mult); positions.append([x, y, size])
        
        positions = filt_util.position_medfilt(np.array(positions), args.medfilt_num)
        previous_frame_tensor = None

        # <<< BẮT ĐẦU CẬP NHẬT LOGIC XỬ LÝ VIDEO >>>
        for idx, frame_path in enumerate(tqdm(frames, desc="Xử lý video frames")):
            img = ipu.imread(str(frame_path), rgb=True) # Đọc ảnh gốc chất lượng cao
            x, y, size = int(positions[idx, 0]), int(positions[idx, 1]), int(positions[idx, 2])
            
            if size > 10:
                T, S, N = 5, 3, 2
                sequence_indices = [np.clip(idx + (t - N) * S, 0, len(frames)-1) for t in range(T)]

                # Crop và resize từng ảnh trong chuỗi
                sequence = []
                all_coords = []
                for i in sequence_indices:
                    seq_img = ipu.imread(str(frames[i]), rgb=True)
                    seq_x, seq_y, seq_size = int(positions[i, 0]), int(positions[i, 1]), int(positions[i, 2])
                    cropped_img, coords = ipu.crop_and_resize_for_model(seq_img, seq_x, seq_y, seq_size, model_input_size=256)
                    if cropped_img is None:
                        cropped_img = np.zeros((256, 256, 3), dtype=np.uint8)
                    sequence.append(cropped_img)
                    all_coords.append(coords)

                center_coords = all_coords[N]
                if center_coords is None:
                    img_result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    ipu.imwrite(str(frames_out_dir / frame_path.name), img_result)
                    continue

                if previous_frame_tensor is None:
                    init_frame_idx = np.clip(idx - 1, 0, len(frames)-1)
                    prev_img_full = ipu.imread(str(frames[init_frame_idx]), rgb=True)
                    prev_x, prev_y, prev_size = int(positions[init_frame_idx, 0]), int(positions[init_frame_idx, 1]), int(positions[init_frame_idx, 2])
                    prev_cropped, _ = ipu.crop_and_resize_for_model(prev_img_full, prev_x, prev_y, prev_size, model_input_size=256)
                    if prev_cropped is None:
                        prev_cropped = np.zeros((256, 256, 3), dtype=np.uint8)
                    previous_frame_tensor = ipu.im2tensor(prev_cropped, bgr2rgb=True, is0_1=False, gpu_id=args.gpu_id)
                
                img_fake, previous_frame_tensor = ipu.run_bvdnet_generator(sequence, previous_frame_tensor, clean_net, gpu_id=args.gpu_id)
                
                mask = ipu.imread(str(masks_dir / frame_path.name), mod='gray')
                img_result = ipu.replace_mosaic(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), img_fake, mask, center_coords, no_feather=args.no_feather)
            else:
                img_result = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            ipu.imwrite(str(frames_out_dir / frame_path.name), img_result)
        # <<< KẾT THÚC CẬP NHẬT LOGIC XỬ LÝ VIDEO >>>
        
        output_dir = Path(args.output_path or (project_root / "output")); output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f"{Path(args.input_path).stem}_cleaned{Path(args.input_path).suffix}"
        ffmpeg_util.build_video_from_frames(str(frames_out_dir / "frame_%06d.png"), str(audio_path) if audio_path.exists() else None, str(output_path), info['fps'], info['resolution'])
        print(f"\nLưu video thành công tại: {output_path}")
    finally:
        if tmp_dir.exists(): shutil.rmtree(tmp_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True); parser.add_argument('--output_path', type=str)
    parser.add_argument('--gpu_id', type=str, default='0'); parser.add_argument('--start_time', type=str)
    parser.add_argument('--end_time', type=str); parser.add_argument('--medfilt_num', type=int, default=11)
    parser.add_argument('--mask_threshold', type=int, default=64); parser.add_argument('--ex_mult', type=float, default=1.1)
    parser.add_argument('--no_feather', action='store_true'); parser.add_argument('--all_mosaic_area', action='store_true')
    args, _ = parser.parse_known_args()
    if args.end_time == "00:00:00": args.end_time = None
    file_ext = Path(args.input_path).suffix.lower()
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']: process_image(args, project_root)
    elif file_ext in ['.mp4', '.mov', '.avi', '.mkv', '.webm']: process_video(args, project_root)
    else: print(f"Định dạng file không được hỗ trợ: {file_ext}"); sys.exit(1)

if __name__ == '__main__':
    main()
