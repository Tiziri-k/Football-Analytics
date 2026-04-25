from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import sys

sys.path.append('../')
from utils import get_center_of_box, get_bbox_width


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        """Run batched inference on a list of frames."""
        batch_size = 4
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_detections = self.model.predict(batch, conf=0.1)
            detections += batch_detections
        return detections

    def get_object_tracks(self, frames):
        """Detect and track players, referees, and the ball across frames."""
        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'ball': [],
            'referees': []
        }

        for frame_num, detection in enumerate(detections):
            cls_name = detection.names
            cls_name_inv = {v: k for k, v in cls_name.items()}

            detection_sv = sv.Detections.from_ultralytics(detection)

            # Reclassify goalkeepers as players before tracking
            for i, cls_id in enumerate(detection_sv.class_id):
                if cls_name[int(cls_id)] == 'goalkeeper':
                    detection_sv.class_id[i] = cls_name_inv['player']

            detection_with_tracks = self.tracker.update_with_detections(detection_sv)

            tracks['players'].append({})
            tracks['ball'].append({})
            tracks['referees'].append({})

            # Store tracked players and referees
            for i in range(len(detection_with_tracks)):
                bbox = detection_with_tracks.xyxy[i].tolist()
                cls_id = int(detection_with_tracks.class_id[i])
                track_id = int(detection_with_tracks.tracker_id[i])

                if cls_id == cls_name_inv['player']:
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}
                elif cls_id == cls_name_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

            # Store ball detections (no tracking, use fixed ID=1)
            for i in range(len(detection_sv)):
                bbox = detection_sv.xyxy[i].tolist()
                cls_id = int(detection_sv.class_id[i])
                if cls_id == cls_name_inv['ball']:
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """Draw an ellipse at the base of a bounding box, with an optional ID label."""
        x_center, _ = get_center_of_box(bbox)
        y2 = int(bbox[3])
        width = get_bbox_width(bbox)

        # Draw base ellipse arc
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(width * 0.35)),
            angle=0,
            startAngle=-45,
            endAngle=225,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # Draw ID label rectangle + text
        if track_id is not None:
            rect_w, rect_h = 40, 20
            x1_rect = x_center - rect_w // 2
            x2_rect = x_center + rect_w // 2
            y1_rect = y2 + 5
            y2_rect = y2 + 5 + rect_h

            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

            # Adjust text x offset based on number of digits
            num_digits = len(str(track_id))
            x_text_offsets = {1: 12, 2: 8, 3: 3}
            x_text = x1_rect + x_text_offsets.get(num_digits, 3)

            cv2.putText(
                frame,
                str(track_id),
                (x_text, y1_rect + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        """Draw a downward-pointing triangle above a bounding box (used for the ball)."""
        x_center, _ = get_center_of_box(bbox)
        y_top = int(bbox[1])

        triangle_points = np.array([
            [x_center, y_top],
            [x_center - 10, y_top - 20],
            [x_center + 10, y_top - 20]
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks):
        """Annotate all frames with player, referee, and ball overlays."""
        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], (0, 0, 255), track_id)

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))

            output_frames.append(frame)

        return output_frames