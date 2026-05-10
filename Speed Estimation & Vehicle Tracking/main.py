from collections import defaultdict, deque
import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np

SOURCES = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0,0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1]
    ]
)

class ViewTransformer:
    def __init__(self, source_points, target_points):
        source_points = np.array(source_points, dtype=np.float32)
        target_points = np.array(target_points, dtype=np.float32)
        self.m = cv2.getPerspectiveTransform(source_points, target_points)
    def transform_points(self, points):
        reshaped_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


if __name__ == "__main__":
    video_path = "D:/Speed Estimation & Vehicle Tracking/data/vehicles.mp4"

    frame_generator = sv.get_video_frames_generator(video_path)
    video_info = sv.VideoInfo.from_video_path(video_path)

    model = YOLO("yolo11x.pt")

    byte_trck = sv.ByteTrack(frame_rate=video_info.fps)

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=thickness,
                                    color_lookup=sv.ColorLookup.TRACK)
    label_annotator = sv.LabelAnnotator(text_thickness=thickness, 
                                        text_scale=text_scale, 
                                        text_position=sv.Position.BOTTOM_CENTER,
                                        color_lookup=sv.ColorLookup.TRACK)
    trace_annotator = sv.TraceAnnotator(thickness=thickness,
                                        trace_length=video_info.fps * 2,
                                        position=sv.Position.BOTTOM_CENTER,
                                        color_lookup=sv.ColorLookup.TRACK)

    polygon = sv.PolygonZone(SOURCES)
    view_transformer = ViewTransformer(SOURCES, TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    for frame in frame_generator:
        results = model(frame)[0]
        detection = sv.Detections.from_ultralytics(results)
        detection = detection[polygon.trigger(detection)]
        detection = byte_trck.update_with_detections(detections=detection)

        points = detection.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_points = view_transformer.transform_points(points).astype(int)
        labels = []

        for track_id, [_, y] in zip(detection.tracker_id, transformed_points):
            coordinates[track_id].append(y)
            if len(coordinates[track_id]) < video_info.fps // 2:
                labels.append(f"ID: {track_id} - Speed: Calculating...")
            else:
                distance = abs(coordinates[track_id][-1] - coordinates[track_id][0])
                time = len(coordinates[track_id]) / video_info.fps
                speed = distance / time
                speed *= 3.6 # Convert m/s to km/h
                labels.append(f"ID: {track_id} - Speed: {speed:.2f} km/h")

        annotated_frame = frame.copy()

        
        
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detection
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detection, labels=labels
        )
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detection
        )

        resized_frame = cv2.resize(annotated_frame, (1280, 720))
        cv2.imshow("Annotated Frame", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
