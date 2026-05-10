import cv2
import os
from utils import get_video_properties, read_video_chunk, make_video_writer
from Trackers import Tracker
from team_assigner import TeamAssigner  
from player_ball_assigner import PlayerBallAssigner
import numpy as np
CHUNK_SIZE = 100  # frames per batch


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

    # Initialize team assigner
    team_assigner = TeamAssigner(num_teams=2)



    while True:
        frames = read_video_chunk(cap, CHUNK_SIZE)
        if not frames:
            break


        chunk_num += 1
        print(f"  Chunk {chunk_num}: {len(frames)} frames", end='', flush=True)

        tracks = tracker.get_object_tracks(frames)
        
        # Interpolate ball positions across the chunk to handle missed detections
        tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

        # Team color assignment based on the first frame of the chunk (can be improved by considering multiple frames)
        team_assigner.assign_team_color(frames[0], tracks['players'][0])
        
        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)

                tracks['players'][frame_num][player_id]['team'] = team
                tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

        # Assign ball to players in each frame
        player_assigner = PlayerBallAssigner() 
        team_ball_controll = []
        for frame_num , player_track in enumerate(tracks['players']):                               
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player_id = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            if assigned_player_id != -1:
                tracks['players'][frame_num][assigned_player_id]['has_ball'] = True
                team_ball_controll.append(tracks['players'][frame_num][assigned_player_id]['team'])
            else :
                team_ball_controll.append(team_ball_controll[-1] if team_ball_controll else 0) 
        
        team_ball_controll = np.array(team_ball_controll)
        
        # Annotate frames with players detection
        annotated = tracker.draw_annotations(frames, tracks,team_ball_controll)

        for frame in annotated:
            writer.write(frame)

        total_frames += len(frames)
        print(f" — done ({total_frames} frames total so far)")
    
    
   
    cap.release()
    writer.release()
    print(f"Finished. Saved {total_frames} frames to: {output_path}")
    return total_frames, tracks

def main(path):
    # Load model once — reused across all videos
    tracker = Tracker('Models/best.pt')
    process_video(path, tracker)


   
if __name__ == "__main__":
    paths = ["input_video/crb_match.mp4"]
    for path in paths:
        main(path)