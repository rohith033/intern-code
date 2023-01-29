import cv2 as cv
import numpy as np
import mediapipe as mp
import time
timer = time.time()
def Angle(a,b,c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c)  
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle
def calAngle(NEAR_LEG,results,height,width):
  if(NEAR_LEG=='Left'):  
    try:
      hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x*height
      hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y*width
      knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x*height
      knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y*width
      ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x*height
      ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y*width
    except AttributeError:
      return 180
  else:
    try:
      hip_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x*height
      hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y*width
      knee_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x*height
      knee_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y*width
      ankle_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x*height
      ankle_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y*width
    except AttributeError:
      return 180
  hip = [hip_x,hip_y]
  knee = [knee_x,knee_y]
  ankle = [ankle_x,ankle_y]
  return Angle(hip,knee,ankle)
cap = cv.VideoCapture('KneeBendVideo (1).mp4')
width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'MP4V') 
out = cv.VideoWriter('stats.mp4', fourcc, 20.0, (width,  height))
# out = cv.VideoWriter('s1.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (640,854))
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
BG_COLOR = (200,200,200)
prev_angle = 180
NEAR_LEG = 'Right'
rep_count = 0 
incorrect_reps = 0
timer_limit = 8
itr = 0
max_rep_time = 0
min_rep_time = 100
prev_angles = []
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      break
    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = pose.process(image)
    image_hight, image_width, _ = image.shape
    depthLeft=0
    depthRight=0
    image.flags.writeable = True
    try:
      depthLeft = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].z 
      depthRight = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].z
    except AttributeError:
      cv.putText(image, f'the leg closer to cam isnt complety visible',(50,200), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
    if(depthLeft<depthRight):
      NEAR_LEG = 'Left'
    angle = calAngle(NEAR_LEG,results,image_hight, image_width)
    prev_angles.append(angle)
    image.flags.writeable = True
    if(len(prev_angles)>5):
      prev_angles.pop(0)
    avarage_angle = np.sum(prev_angles) / len(prev_angles)
    if(avarage_angle<140 and prev_angle>140):
      timer = time.time()
    if avarage_angle > 140 and prev_angle < 140:
        if time.time() - timer < timer_limit and time.time() - timer > 1:
          incorrect_reps=incorrect_reps+1
          cv.putText(image, f'keep your knee bent',(50,180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0, 255), 1, cv.LINE_AA)
          timer = time.time()
    if time.time() - timer >= timer_limit and avarage_angle > 160:
            max_rep_time = max(max_rep_time, time.time() - timer)
            min_rep_time = min(min_rep_time, time.time() - timer)
            timer = time.time()
            rep_count += 1 
    cv.putText(image, f'no of reps {rep_count}',(50,70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
    cv.putText(image, f'cur angle {avarage_angle}',(50,90), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
    cv.putText(image, f'prev_angle {prev_angle}',(50,110), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
    cv.putText(image, f'timer {timer}',(50,130), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
    cv.putText(image, f'incorrect reps {incorrect_reps}',(50,150), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
    cv.putText(image, f'max_time_for_a_rep {max_rep_time}',(500,50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)
    cv.putText(image, f'min_time_for_a_rep {min_rep_time}',(500,70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0, 0), 1, cv.LINE_AA)

    out.write(image)
    cv.imshow('MediaPipePose', image)
    prev_angle = avarage_angle
    if cv.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv.destroyAllWindows()