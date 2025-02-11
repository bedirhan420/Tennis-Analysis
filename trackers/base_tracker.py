from ultralytics import YOLO
import pickle

class BaseTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                detections = pickle.load(f)
            #print(f"Readed detections : {detections}")
            return detections

        for frame in frames:
            detection = self.detect_frame(frame)
            detections.append(detection)
        #print(f"Writed detections : {detections}")
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(detections, f, protocol=pickle.HIGHEST_PROTOCOL)
        return detections

    def detect_frame(self, frame):
        raise NotImplementedError("Subclasses must implement detect_frame method.")

    def draw_boxes(self, video_frames, detections):
        output_video_frames = []
        for frame, detection_dict in zip(video_frames, detections):
            frame = self.draw_boxes_on_frame(frame, detection_dict)
            output_video_frames.append(frame)
        return output_video_frames

    def draw_boxes_on_frame(self, frame, detection_dict):
        raise NotImplementedError("Subclasses must implement draw_boxes_on_frame method.")