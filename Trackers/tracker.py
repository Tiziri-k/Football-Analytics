from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import sys
import pandas as pd

sys.path.append('../')
from utils import get_center_of_box, get_bbox_width


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]  
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        # interpolate missing values (0s) using linear interpolation, then fill any remaining NaNs with 0
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1:{'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions


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
            x_text = x1_rect + x_text_offsets.get(num_digits, 2)

            cv2.putText(
                frame,
                str(track_id),
                (x_text, y1_rect + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
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

    
    def draw_team_ball_control(self, frame, frame_num,team_ball_controll):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255,255,255), cv2.FILLED)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        team_ball_controll_till_frame = team_ball_controll[:frame_num+1]
        team_0_num_frames = team_ball_controll_till_frame[team_ball_controll_till_frame == 1].shape[0]
        team_1_num_frames = team_ball_controll_till_frame[team_ball_controll_till_frame == 2].shape[0]
        total_frames = team_0_num_frames + team_1_num_frames
        if total_frames > 0:
            team1 = (team_0_num_frames / total_frames) * 100
            team2 = (team_1_num_frames / total_frames) * 100
        else:
            team1 = 0
            team2 = 0
        cv2.putText(frame, f'Team 1 Ball Control: {team1:.1f}%', (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f'Team 2 Ball Control: {team2:.1f}%', (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        return frame

    def draw_annotations(self, video_frames, tracks,team_ball_controll):
        """Annotate all frames with player, referee, and ball overlays."""
        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]

            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))  # Default to red if team color not assigned
                frame = self.draw_ellipse(frame, player['bbox'], color, track_id)
                if player.get('has_ball'):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 0, 255))
            
            
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 255, 255))

            for _, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))
            
            # Draw team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_controll)
            output_frames.append(frame)

        return output_frames