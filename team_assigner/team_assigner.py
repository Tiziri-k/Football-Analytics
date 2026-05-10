from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self, num_teams=2):
        self.num_teams = num_teams
        self.team_colors = {}
        self.player_team_dict = {}
    
    def get_clustering_model(self, image):
        reshaped_image = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.num_teams, init='k-means++', n_init=1,random_state=42)
        kmeans.fit(reshaped_image)
        return kmeans
    
    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_image = image[0:int(image.shape[0]/2), :]
        kmeans = self.get_clustering_model(top_half_image)
        # Get the cluster centers (team colors)
        lables = kmeans.labels_

        #Reshape the labels to match the original image shape
        clustered_image  = lables.reshape (top_half_image.shape[0], top_half_image.shape[1])
        corner_cluster = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_cluster), key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster
        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color
    
    def assign_team_color(self,frame,player_detections):
        
        player_colors = []
        for _ , player_detection in player_detections.items():
            bbox = player_detection['bbox']
            color = self.get_player_color(frame,bbox)
            player_colors.append(color)
        kmeans = KMeans(n_clusters=self.num_teams, init='k-means++', n_init=1,random_state=42)
        kmeans.fit(player_colors)
        
        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        
    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        else:
            player_color = self.get_player_color(frame,player_bbox)
            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
            team_id +=1
            self.player_team_dict[player_id] = team_id
            return team_id