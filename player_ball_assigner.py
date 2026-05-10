import numpy as np
from utils import get_center_of_box

class PlayerBallAssigner:
    def __init__(self, max_distance=70):
        self.max_distance = max_distance

    def assign_ball_to_player(self, player_track, ball_bbox):
        ball_center = get_center_of_box(ball_bbox)

        min_distance = float('inf')
        assigned_player = -1

        for player_id, track in player_track.items():
            player_bbox = track['bbox']
            player_center = get_center_of_box(player_bbox)

            distance = np.linalg.norm(np.array(ball_center) - np.array(player_center))

            if distance < self.max_distance and distance < min_distance:
                min_distance = distance
                assigned_player = player_id

        return assigned_player