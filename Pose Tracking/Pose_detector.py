import cv2
import mediapipe as mp
import numpy as np
import os
from config import *
import json
    
class Pose_detector():
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.connection_list = frozenset(landmark_lists)  # Assuming squat_landmark_lists is defined in config.py

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # first point
        b = np.array(b)  # middle point
        c = np.array(c)  # final point

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def check_squat(self, angle_knee, stage):
        if angle_knee > 169:
            stage = "up"

        if angle_knee <= 100 and stage == 'reduce':
            stage = "down"
            return True, stage
        
        if 160 < angle_knee < 169 and stage == 'up':
            stage = "reduce"
            return True, stage

        return False, stage

    def take_squat_land_mark(self, landmarks):
        hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        angle_knee = round(self.calculate_angle(hip, knee, ankle))

        return angle_knee

    def check_pushup(self, angle_elbow, stage):
        if angle_elbow > 120:
            stage = "up"

        if angle_elbow <= 80 and stage == 'reduce':
            stage = "down"
            print("down")
            return True, stage
    
        if 98 < angle_elbow < 120 and stage == 'up':
            stage = "reduce"
            print("reduce")
            return True, stage

        return False, stage
    
    
    

    def take_pushup_land_mark(self, landmarks):
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        angle_elbow = round(self.calculate_angle(shoulder, elbow, wrist))

        return angle_elbow

    def calculate_iou(self, interval_a, interval_b):
        start_a, end_a = interval_a
        start_b, end_b = interval_b

        # Check if pred_interval (interval_b) is completely within label interval (interval_a)
        if start_a <= start_b and end_b <= end_a:
            return 1.0

        intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
        union = max(end_a, end_b) - min(start_a, start_b)

        iou = intersection / union if union != 0 else 0
        return iou
    
    def argmax(self, lst):
        return max(range(len(lst)), key=lambda i: lst[i])
    
    def evaluate(self, label_dir):
        avg_acc = 0
        for file_name in os.listdir(label_dir):
            acc = 0
            label_file = os.path.join(label_dir, file_name)
            pred_file = label_file.replace("label", "pred")
            print(file_name)
            with open(label_file, 'r') as f, open(pred_file, 'r') as f_pred:
                
                label = json.load(f)
                pred = json.load(f_pred)
                for interval in label["label"]:
                    ious = [self.calculate_iou(interval, pred_interval) for pred_interval in pred["label"]]
                    print(ious)
                    if ious:  # Ensure ious is not empty
                        indx = self.argmax(ious)
                        if ious[indx] > 0.9:
                            acc += 1
            avg_acc += acc / label["count"]
        print(avg_acc / len(os.listdir(label_dir)))
                    
    
    def inference(self, action, video_path=None):
        
        
        if not video_path:
            video_path = 0
        else:
            output_path = video_path.replace("video", "output")
            pred_path = video_path.replace("video", "pred").replace("avi", "json")
            list_frame = []

        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

        # Get frame properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if video_path != 0:
        # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        counter = 0
        stage = None

        try:
            with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                i = -1
                while cap.isOpened():
                    i += 1
                    if i % FRAME_STEP != 0:
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        break

                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    try:
                        landmarks = results.pose_landmarks.landmark
                    except Exception as e:
                        continue

                    if action == 'squat':
                        angle_knee = self.take_squat_land_mark(landmarks)
                        status, stage = self.check_squat(angle_knee, stage)                            
                    elif action == 'pushup':
                        angle_elbow = self.take_pushup_land_mark(landmarks)
                        print(angle_elbow)
                        status, stage = self.check_pushup(angle_elbow, stage)

                    print(status, stage)
                    
                    if status and stage == "down": # reduce -> up
                        counter += 1
                        frame_end = i
                        list_frame.append([frame_start, frame_end])
                        

                    elif status and stage == "reduce": # up -> reduce
                        frame_start = i
                            
                
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.connection_list,
                                                   self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                                   self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                                   )
                    cv2.putText(image, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

                    # Write the frame to the output video file
                    out.write(image)

                    cv2.imshow('Mediapipe Feed', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

                cap.release()
                out.release()
                cv2.destroyAllWindows()
        except Exception as e:
            print(e)
        
        
        if video_path != 0:
            with open(pred_path, "w+") as f:
                data = {"count": counter, "label": list_frame}
                json.dump(data, f)
                
                
