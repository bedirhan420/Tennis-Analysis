import cv2
from .base_tracker import BaseTracker
import sys
sys.path.append("../")
from utils import measure_distance,get_center_of_bbox

class PlayerTracker(BaseTracker):
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names
        player_dict = {}

        for box in results.boxes:
            if box.id is not None:  # `box.id` kontrol ediliyor
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]
                object_class_id = box.cls.tolist()[0]
                object_class_name = id_name_dict[object_class_id]
                if object_class_name == "person":
                    player_dict[track_id] = result
        # print(f"player dict : {player_dict}")
        return player_dict


    def draw_boxes_on_frame(self, frame, detection_dict):
        for track_id, box in detection_dict.items():
            x1, y1, x2, y2 = box
            cv2.putText(frame, f"Player ID: {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        return frame
    
    def choose_and_filter_players(self, court_keypoints, player_detections):
        #print(f" player detections : {player_detections} ")
        player_detections_first_frame = next(
            (frame for frame in player_detections if frame),None
        )
        #print(f" player detections first frame : {player_detections_first_frame} ")
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        #print(f" court keypoints : {court_keypoints} ")

        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)
            #print(f" player center : {player_center} ")

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
       # print(f" distances : {distances} ")

        distances.sort(key = lambda x: x[1])
        #print(f"sorted distances : {distances} ")

        chosen_players = [distances[0][0], distances[1][0]]
        #print(f"chosen players : {chosen_players} ")
        return chosen_players