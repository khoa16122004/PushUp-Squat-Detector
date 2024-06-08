from Pose_detector import Pose_detector
from config import *


a = Pose_detector()
if mode == "video":
    a.inference(action, video_path)

else:
    a.inference(action)
