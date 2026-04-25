import cv2
import os
from utils import get_video_properties, read_video_chunk, make_video_writer
from Trackers import Tracker


CHUNK_SIZE = 100  # frames per batch — lower this if you still run out of RAM


def process_video(path, tracker):
    print(f"\nProcessing: {path}")

    fps, width, height = get_video_properties(path)

    input_filename = os.path.splitext(os.path.basename(path))[0]
    output_path = f'output_videos/{input_filename}_output.mp4'

    writer = make_video_writer(output_path, fps, width, height)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    total_frames = 0
    chunk_num = 0

    while True:
        frames = read_video_chunk(cap, CHUNK_SIZE)
        if not frames:
            break

        chunk_num += 1
        print(f"  Chunk {chunk_num}: {len(frames)} frames", end='', flush=True)

        tracks = tracker.get_object_tracks(frames)
        annotated = tracker.draw_annotations(frames, tracks)

        for frame in annotated:
            writer.write(frame)

        total_frames += len(frames)
        print(f" — done ({total_frames} frames total so far)")

    cap.release()
    writer.release()
    print(f"Finished. Saved {total_frames} frames to: {output_path}")


def main(paths):
    # Load model once — reused across all videos
    tracker = Tracker('Models/best.pt')

    for path in paths:
        process_video(path, tracker)


if __name__ == "__main__":
    paths = ["input_video/crb_match.mp4"]
    main(paths)