
import os
import json
import subprocess

def run_ffmpeg_command(args):
    """
    Thực thi một lệnh ffmpeg và xử lý output.
    """
    # Chuyển đổi tất cả các đối số thành chuỗi để tránh lỗi
    args = [str(arg) for arg in args]
    try:
        # Sử dụng subprocess.run để có kiểm soát tốt hơn
        process = subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )
        return process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi thực thi lệnh FFmpeg: {' '.join(args)}")
        print(f"Stderr: {e.stderr}")
        raise  # Ném lại lỗi để có thể xử lý ở cấp cao hơn
    except FileNotFoundError:
        print("Lỗi: Lệnh 'ffmpeg' hoặc 'ffprobe' không được tìm thấy. Hãy chắc chắn rằng FFmpeg đã được cài đặt và có trong PATH của hệ thống.")
        raise

def get_video_info(video_path):
    """
    Lấy thông tin của video như fps, duration, resolution bằng ffprobe.
    Trả về một dictionary chứa thông tin.
    """
    args = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    try:
        out_string = run_ffmpeg_command(args)
        infos = json.loads(out_string)

        # Tìm video stream
        video_stream = None
        for stream in infos['streams']:
            if stream['codec_type'] == 'video':
                video_stream = stream
                break

        if not video_stream:
            raise ValueError("Không tìm thấy video stream trong file.")

        fps = eval(video_stream.get('avg_frame_rate', '0/1'))
        duration = float(infos['format'].get('duration', 0))
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))

        return {
            'fps': fps,
            'duration': duration,
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}"
        }
    except Exception as e:
        print(f"Không thể lấy thông tin từ video: {video_path}. Lỗi: {e}")
        return None


def extract_frames(video_path, output_pattern, fps=None, start_time=None, end_time=None):
    """
    Trích xuất các khung hình từ video.
    """
    args = ['ffmpeg', '-y']
    if start_time:
        args.extend(['-ss', start_time])
    if end_time:
        # ffmpeg dùng -t (duration) hoặc -to. -to an toàn hơn.
        args.extend(['-to', end_time])

    args.extend(['-i', video_path])
    if fps:
        args.extend(['-r', str(fps)])

    args.extend(['-q:v', '2', output_pattern]) # -q:v 2 cho chất lượng tốt
    run_ffmpeg_command(args)
    print(f"Đã trích xuất frames từ '{video_path}' vào pattern '{output_pattern}'")


def extract_audio(video_path, audio_path, start_time=None, end_time=None):
    """
    Trích xuất file âm thanh từ video.
    """
    if os.path.exists(audio_path):
        os.remove(audio_path)

    args = ['ffmpeg', '-y']
    if start_time:
        args.extend(['-ss', start_time])
    if end_time:
        args.extend(['-to', end_time])

    args.extend([
        '-i', video_path,
        '-vn',          # No video
        '-acodec', 'copy' # Sao chép audio codec để giữ nguyên chất lượng
    ])
    args.append(audio_path)

    try:
        run_ffmpeg_command(args)
        print(f"Đã trích xuất audio từ '{video_path}' sang '{audio_path}'")
        return True
    except subprocess.CalledProcessError:
        print(f"Không có audio stream trong '{video_path}' hoặc lỗi. Bỏ qua trích xuất audio.")
        return False


def build_video_from_frames(frame_pattern, audio_path, output_path, fps, resolution):
    """
    Ghép các frame và audio thành video hoàn chỉnh.
    """
    args = [
        'ffmpeg', '-y',
        '-r', str(fps),
        '-f', 'image2',
        '-s', resolution,
        '-i', frame_pattern
    ]

    has_audio = audio_path and os.path.exists(audio_path)
    if has_audio:
        args.extend(['-i', audio_path])
        args.extend(['-c:a', 'aac', '-b:a', '192k']) # Encode audio

    args.extend([
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p' # Định dạng pixel tương thích rộng rãi
    ])

    if has_audio:
        args.extend(['-shortest']) # Kết thúc video khi stream ngắn nhất (audio/video) kết thúc

    args.append(output_path)
    run_ffmpeg_command(args)
    print(f"Đã tạo video '{output_path}' thành công.")

def cut_video_segment(input_path, output_path, start_time=None, end_time=None, to_time=None):
    """
    Cắt một đoạn video.
    """
    args = ['ffmpeg', '-y']

    if start_time is not None:
        args.extend(['-ss', str(start_time)])

    if to_time is not None:
         args.extend(['-to', str(to_time)])

    args.extend(['-i', input_path])

    if end_time is not None and to_time is None: # end_time is treated as duration if -to is not present
        args.extend(['-t', str(end_time)])

    args.extend(['-c', 'copy', output_path])

    run_ffmpeg_command(args)
    print(f"Đã cắt video '{input_path}' sang '{output_path}'.")


def split_video_into_segments(input_path, output_pattern, segment_time):
    """
    Chia video thành các đoạn nhỏ theo thời gian.
    """
    args = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-c', 'copy',
        '-map', '0',
        '-segment_time', str(segment_time),
        '-f', 'segment',
        '-reset_timestamps', '1',
        output_pattern
    ]
    run_ffmpeg_command(args)
    print(f"Đã chia video '{input_path}' thành các segments.")


def merge_videos(file_list_path, output_path):
    """
    Ghép các video từ một file text chứa danh sách.
    """
    args = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', file_list_path,
        '-c', 'copy',
        output_path
    ]
    run_ffmpeg_command(args)
    print(f"Đã ghép các video và lưu tại '{output_path}'.")

