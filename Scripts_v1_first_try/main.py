"""Football Player & Ball Detection Pipeline
==========================================
Detects players, ball, referee from match video using YOLOv8x.
Outputs a tracking CSV ready for heatmaps, physical reports,
and defensive shape analysis.

Install dependencies first:
    pip install ultralytics opencv-python pandas numpy supervision
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.cluster import KMeans
import supervision as sv
import os


# ─────────────────────────────────────────
# CONFIG — change these to match your setup
# ─────────────────────────────────────────

VIDEO_PATH   = "input_video\crb_match.mp4"       # your downloaded LP1 video
OUTPUT_VIDEO = "crb_tracked.mp4"     # annotated output video
OUTPUT_CSV   = "crb_tracking.csv"    # tracking data for analysis

# Real pitch dimensions in meters (standard football pitch)
PITCH_WIDTH_M  = 105.0
PITCH_HEIGHT_M = 68.0

# Detection confidence threshold — lower = detect more (but more noise)
CONF_THRESHOLD = 0.3

# Sprint speed thresholds in m/s (FIFA standard)
WALK_MS      = 2.0
JOG_MS       = 4.0
RUN_MS       = 5.5
HIGH_RUN_MS  = 7.0
SPRINT_MS    = 7.0


# ─────────────────────────────────────────
# STEP 1 — LOAD MODEL
# ─────────────────────────────────────────

def load_model():
    """
    Load YOLOv8x pretrained on COCO.
    First run downloads weights automatically (~130MB).
    
    For better ball detection, swap with a football-specific model:
        model = YOLO("football-player-detection.pt")
    """
    print("[1/5] Loading YOLOv8x model...")
    model = YOLO("yolov8x.pt")
    return model


# ─────────────────────────────────────────
# STEP 2 — TEAM COLOR SEPARATION (KMeans)
# ─────────────────────────────────────────

def get_player_color(frame, bbox):
    """
    Crop the player bounding box, focus on the jersey area (top 50%),
    and return the dominant color as RGB.
    Used by KMeans to separate teams by jersey color.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Crop to bounding box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.array([0, 0, 0])
    
    # Focus on jersey (top 50% of crop, avoid shorts and pitch)
    jersey_h = int(crop.shape[0] * 0.5)
    jersey   = crop[:jersey_h, :]
    
    # Reshape to list of pixels
    pixels = jersey.reshape(-1, 3).astype(np.float32)
    
    if len(pixels) < 10:
        return np.array([0, 0, 0])
    
    # KMeans to find dominant color (ignore background with 2 clusters)
    kmeans = KMeans(n_clusters=2, n_init=5, random_state=42)
    kmeans.fit(pixels)
    
    # Return the cluster center with more pixels (dominant color)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_idx   = labels[np.argmax(counts)]
    return kmeans.cluster_centers_[dominant_idx]


def assign_teams(player_colors, n_teams=2):
    """
    Given a list of jersey colors, use KMeans to assign each
    player to team 0 or team 1.
    Returns array of team IDs.
    """
    if len(player_colors) < n_teams:
        return np.zeros(len(player_colors), dtype=int)
    
    colors_arr = np.array(player_colors, dtype=np.float32)
    kmeans     = KMeans(n_clusters=n_teams, n_init=10, random_state=42)
    team_ids   = kmeans.fit_predict(colors_arr)
    return team_ids


# ─────────────────────────────────────────
# STEP 3 — PERSPECTIVE TRANSFORM
# ─────────────────────────────────────────

def get_perspective_transform(frame_width, frame_height):
    """
    Map pixel coordinates to real-world pitch coordinates (meters).
    
    You need to manually set the 4 corner pixel coordinates of the
    pitch in your specific video. These are approximate defaults.
    
    HOW TO FIND YOUR CORNERS:
        1. Open your video in VLC
        2. Pause on a frame where you see the full pitch
        3. Note the pixel (x, y) of each corner of the pitch
        4. Replace the values below
    """
    # Source: pixel coordinates of pitch corners in your video
    # Order: top-left, top-right, bottom-right, bottom-left
    src_points = np.float32([
        [100, 80],                          # top-left corner of pitch
        [frame_width - 100, 80],            # top-right
        [frame_width - 50, frame_height - 50],  # bottom-right
        [50, frame_height - 50]             # bottom-left
    ])
    
    # Destination: real pitch dimensions in meters (scaled to pixels)
    scale = 6  # 1 meter = 6 pixels in output map
    dst_points = np.float32([
        [0, 0],
        [int(PITCH_WIDTH_M * scale), 0],
        [int(PITCH_WIDTH_M * scale), int(PITCH_HEIGHT_M * scale)],
        [0, int(PITCH_HEIGHT_M * scale)]
    ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return M, scale


def pixel_to_meters(px, py, M, scale):
    """
    Convert pixel coordinates to real-world meters using
    the perspective transform matrix M.
    """
    point = np.float32([[[px, py]]])
    transformed = cv2.perspectiveTransform(point, M)
    real_x = transformed[0][0][0] / scale
    real_y = transformed[0][0][1] / scale
    return real_x, real_y


# ─────────────────────────────────────────
# STEP 4 — MAIN TRACKING LOOP
# ─────────────────────────────────────────

def run_tracking_pipeline(model, video_path, output_video, output_csv):
    """
    Main loop: read video frame by frame, detect objects,
    assign teams, calculate speed, save tracking data.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    # Video properties
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[2/5] Video loaded: {frame_width}x{frame_height} @ {fps:.1f}fps")
    print(f"      Total frames: {total_frames} ({total_frames/fps/60:.1f} minutes)")
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_video, fourcc, fps,
                             (frame_width, frame_height))
    
    # Perspective transform
    M, scale = get_perspective_transform(frame_width, frame_height)
    
    # Tracking state — stores last known position per track_id
    prev_positions = {}   # {track_id: (real_x, real_y)}
    
    # Results accumulator
    records = []
    
    # Supervision tracker for consistent track IDs across frames
    tracker = sv.ByteTrack()
    
    print("[3/5] Running detection + tracking...")
    frame_num = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        
        # Progress update every 100 frames
        if frame_num % 100 == 0:
            pct = frame_num / total_frames * 100
            print(f"      Frame {frame_num}/{total_frames} ({pct:.1f}%)")
        
        # ── Run YOLO detection ──
        results = model(frame, conf=CONF_THRESHOLD, classes=[0], verbose=False)[0]
        
        # Convert to supervision format for tracking
        detections = sv.Detections.from_ultralytics(results)
        
        # Update tracker — assigns consistent track_id across frames
        detections = tracker.update_with_detections(detections)
        
        if len(detections) == 0:
            out.write(frame)
            continue
        
        # ── Team color assignment ──
        player_colors = []
        for bbox in detections.xyxy:
            color = get_player_color(frame, bbox)
            player_colors.append(color)
        
        team_ids = assign_teams(player_colors)
        
        # ── Process each detected player ──
        annotated = frame.copy()
        
        for i, (bbox, track_id) in enumerate(
            zip(detections.xyxy, detections.tracker_id)
        ):
            if track_id is None:
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Player center in pixels
            center_px = int((x1 + x2) / 2)
            center_py = int(y2)  # use feet position (bottom of bbox)
            
            # Convert to real-world meters
            real_x, real_y = pixel_to_meters(center_px, center_py, M, scale)
            
            # Clamp to pitch boundaries
            real_x = max(0, min(real_x, PITCH_WIDTH_M))
            real_y = max(0, min(real_y, PITCH_HEIGHT_M))
            
            # ── Calculate speed (m/s) ──
            speed_ms = 0.0
            if track_id in prev_positions:
                prev_x, prev_y = prev_positions[track_id]
                dx = real_x - prev_x
                dy = real_y - prev_y
                distance_m = np.sqrt(dx**2 + dy**2)
                speed_ms   = distance_m * fps  # distance per frame × fps
                
                # Clamp unrealistic speeds (> 12 m/s = ~43 km/h)
                speed_ms = min(speed_ms, 12.0)
            
            prev_positions[track_id] = (real_x, real_y)
            
            # ── Classify movement ──
            if speed_ms < WALK_MS:
                movement = "walk"
            elif speed_ms < JOG_MS:
                movement = "jog"
            elif speed_ms < RUN_MS:
                movement = "run"
            elif speed_ms < HIGH_RUN_MS:
                movement = "high_run"
            else:
                movement = "sprint"
            
            # ── Team assignment ──
            team_id = int(team_ids[i]) if i < len(team_ids) else 0
            
            # ── Store record ──
            records.append({
                "frame"      : frame_num,
                "time_s"     : round(frame_num / fps, 2),
                "track_id"   : int(track_id),
                "team"       : team_id,
                "pixel_x"    : center_px,
                "pixel_y"    : center_py,
                "real_x"     : round(real_x, 2),    # meters from left
                "real_y"     : round(real_y, 2),    # meters from top
                "speed_ms"   : round(speed_ms, 3),
                "speed_kmh"  : round(speed_ms * 3.6, 2),
                "movement"   : movement,
            })
            
            # ── Draw annotation on frame ──
            color = (0, 0, 255) if team_id == 0 else (255, 0, 0)  # red/blue
            cv2.rectangle(annotated,
                          (int(x1), int(y1)), (int(x2), int(y2)),
                          color, 2)
            cv2.putText(annotated,
                        f"#{track_id} {speed_ms*3.6:.0f}km/h",
                        (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        out.write(annotated)
    
    cap.release()
    out.release()
    
    print(f"[4/5] Detection complete. {len(records)} records captured.")
    return records


# ─────────────────────────────────────────
# STEP 5 — SAVE TRACKING CSV
# ─────────────────────────────────────────

def save_tracking_data(records, output_csv):
    """
    Save all tracking records to CSV.
    This CSV feeds directly into your heatmap, physical report,
    and defensive shape scripts.
    """
    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    
    print(f"[5/5] Tracking data saved to: {output_csv}")
    print(f"\n── Summary ──")
    print(f"Total frames processed : {df['frame'].max()}")
    print(f"Unique players tracked : {df['track_id'].nunique()}")
    print(f"Duration               : {df['time_s'].max()/60:.1f} minutes")
    print(f"Team 0 detections      : {(df['team']==0).sum()}")
    print(f"Team 1 detections      : {(df['team']==1).sum()}")
    print(f"\nSpeed stats:")
    print(df.groupby("movement")["track_id"].count().rename("count"))
    
    return df


# ─────────────────────────────────────────
# STEP 6 — QUICK SANITY CHECK
# ─────────────────────────────────────────

def sanity_check(df):
    """
    Quick checks to verify your tracking data is sensible
    before feeding it to the analysis scripts.
    """
    print("\n── Sanity Check ──")
    
    # Check 1: are positions within pitch bounds?
    out_of_bounds = df[
        (df["real_x"] < 0) | (df["real_x"] > PITCH_WIDTH_M) |
        (df["real_y"] < 0) | (df["real_y"] > PITCH_HEIGHT_M)
    ]
    pct_oob = len(out_of_bounds) / len(df) * 100
    print(f"Out-of-bounds positions: {pct_oob:.1f}%  (should be < 5%)")
    
    # Check 2: are speeds realistic?
    avg_speed = df["speed_ms"].mean()
    print(f"Average speed: {avg_speed*3.6:.1f} km/h  (should be 5–12 km/h)")
    
    # Check 3: are track IDs stable?
    avg_frames_per_player = df.groupby("track_id")["frame"].count().mean()
    print(f"Avg frames per player: {avg_frames_per_player:.0f}  (should be > 50)")
    
    if pct_oob > 10:
        print("\n⚠ Warning: many out-of-bounds positions.")
        print("  → Adjust src_points in get_perspective_transform()")
    
    if avg_speed * 3.6 > 20:
        print("\n⚠ Warning: average speed seems too high.")
        print("  → Check your scale value in get_perspective_transform()")


# ─────────────────────────────────────────
# MAIN — run everything
# ─────────────────────────────────────────

if __name__ == "__main__":
    
    # Check video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Video not found: {VIDEO_PATH}")
        print("   Download a CRB match first:")
        print("   yt-dlp -f 'best[ext=mp4]' 'YOUR_YOUTUBE_URL'")
        print("   Then rename it to crb_match.mp4")
        exit(1)
    
    # Run pipeline
    model   = load_model()
    records = run_tracking_pipeline(model, VIDEO_PATH, OUTPUT_VIDEO, OUTPUT_CSV)
    df      = save_tracking_data(records, OUTPUT_CSV)
    sanity_check(df)
    
    print(f"\n✓ Done. Your files:")
    print(f"  Annotated video : {OUTPUT_VIDEO}")
    print(f"  Tracking CSV    : {OUTPUT_CSV}")
    print(f"\nNext step: feed {OUTPUT_CSV} into heatmap.py / physical_report.py")