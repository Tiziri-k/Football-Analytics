import cv2
import os


def get_video_properties(video_path):
    """Return (fps, width, height) for a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 24
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height


def frame_generator(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def read_video_chunk(cap, chunk_size):
    frames = []
    for _ in range(chunk_size):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def make_video_writer(output_path, fps, width, height):
    """Create and return a cv2.VideoWriter, raising clearly if it fails."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    ext = output_path.lower().rsplit('.', 1)[-1]
    fourcc_map = {'mp4': 'mp4v', 'avi': 'XVID'}
    fourcc = cv2.VideoWriter_fourcc(*fourcc_map.get(ext, 'mp4v'))

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {output_path}")
    return writer


# ── Kept for backward compatibility ──────────────────────────────────────────

def read_video(video_path):
  
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from: {video_path}")
    return frames


def save_video(output_video_frames, output_video_path):
    """Write a list of frames to disk."""
    if not output_video_frames:
        raise ValueError("No frames to save.")

    h, w = output_video_frames[0].shape[:2]
    writer = make_video_writer(output_video_path, fps=24, width=w, height=h)
    for frame in output_video_frames:
        writer.write(frame)
    writer.release()
    print(f"Saved: {output_video_path}")