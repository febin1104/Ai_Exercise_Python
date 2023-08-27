import cv2
import mediapipe as mp
import numpy as np
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def camera():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    counter = 0
    count1=0
    count2=0
    count3=0
    count4=0
    stage = None
    key=None
    exercise="AI EXERCISE"
    calories=0
    calo1=0
    calo2=0
    calo3=0
    calo4=0
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    cv2.namedWindow('AI EXERCISE', cv2.WINDOW_NORMAL)
    path=r"D:\FEBIN RAJAN\pythonProject\Exercise\resources\AI_EXERCISE1.jpg"
    img_rgb=cv2.imread(path,1)
    cv2.imshow('AI EXERCISE',img_rgb)
    cv2.waitKey(0)
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            # Make detection
            results = pose.process(image)
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle
                angle1 = calculate_angle(shoulder, elbow, wrist)
                angle2 = calculate_angle(hip, knee, ankle)
                angle3 = calculate_angle(shoulder, hip, knee)
                angle4 = calculate_angle(shoulder, elbow, wrist)
            except:
                pass
            if key == 1 :
                cv2.putText(image, str(angle1),
                            tuple(np.multiply(elbow, [1280, 720]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle1 > 160:
                    stage = "up"
                if angle1 < 70 and stage == 'up':
                    stage = "down"
                    counter += 1
                    count1+=1
                    calo1+=0.36
                    calories+=0.36
                    calories=round(calories,2)
                    print(counter)
            elif key == 2 :
                # Visualize angle
                cv2.putText(image, str(angle2),
                            tuple(np.multiply(knee, [1280, 720]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle2 > 160:
                    stage = "up"
                if angle2 < 70 and stage == 'up':
                    stage = "down"
                    counter += 1
                    count2 += 1
                    calo2 += 0.32
                    calories += 0.32
                    calories = round(calories, 2)
                    print(counter)
            elif key == 3:
                # Visualize angle
                cv2.putText(image, str(angle3),
                            tuple(np.multiply(hip, [1280, 720]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle3 > 105:
                    stage = "down"
                if angle3 < 55 and stage == 'down':
                    stage = "up"
                    counter += 1
                    count3 += 1
                    calo3 += 0.25
                    calories += 0.25
                    calories = round(calories, 2)
                    print(counter)
            elif key == 4:
                # Visualize angle
                cv2.putText(image, str(angle4),
                            tuple(np.multiply(elbow, [1280, 720]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )

                # Curl counter logic
                if angle4 > 160:
                    stage = "down"
                if angle4 < 30 and stage == 'down':
                    stage = "up"
                    counter += 1
                    count4 += 1
                    calo4 += 0.9
                    calories += 0.9
                    calories = round(calories, 2)
                    print(counter)

            #Exercise Data
            cv2.putText(image, 'Exercise: ', (5,25), cv2.FONT_HERSHEY_PLAIN, 1.5,
                        (204, 0, 0), 2)
            cv2.putText(image, exercise,
                        (125, 25),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (204, 0, 0),2, cv2.LINE_AA)
            # Counter Data
            cv2.putText(image, 'Counter: ', (5, 45), cv2.FONT_HERSHEY_PLAIN, 1.5,
                        (204, 0, 0), 2)
            cv2.putText(image, str(counter),
                        (125, 45),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (204, 0, 0), 2, cv2.LINE_AA)
            # Stage Data
            cv2.putText(image, 'Stage: ', (5, 65), cv2.FONT_HERSHEY_PLAIN, 1.5,
                        (204, 0, 0), 2)
            cv2.putText(image, stage,
                        (125, 65),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (204, 0, 0), 2, cv2.LINE_AA)
            # Calories Data
            cv2.putText(image, 'Calories: ', (5, 85), cv2.FONT_HERSHEY_PLAIN, 1.5,
                        (204, 0, 0), 2)
            cv2.putText(image, str(calories),
                        (125, 85),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (204, 0, 0), 2, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2, circle_radius=2)
                                      )
            cv2.putText(image, 'Press 1: Push Up', (5, 590), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)
            cv2.putText(image, 'Press 2: Squat', (5, 620), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)
            cv2.putText(image, 'Press 3: Sit Up', (5, 650), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)
            cv2.putText(image, 'Press 4: Biceps', (5, 680), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)
            cv2.putText(image, 'Press q: Exit', (5,710), cv2.FONT_HERSHEY_PLAIN, 2,
                        (0, 255, 0), 2)
            cv2.imshow('AI EXERCISE', image)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('1'):
                key = 1
                counter = 0
                stage = None
                exercise="Push Up"
            elif k == ord('2'):
                key = 2
                counter = 0
                stage = None
                exercise = "Squats"
            elif k == ord('3'):
                key = 3
                counter = 0
                stage = None
                exercise = "Sit Up"
            elif k == ord('4'):
                key = 4
                counter = 0
                stage = None
                exercise = "Bicepss"
            elif k == ord('q'):
                break
        cap.release()
        path=r"D:\FEBIN RAJAN\pythonProject\Exercise\resources\EXERCISE_CHART1.jpg"
        img_rgb=cv2.imread(path,1)
        cv2.putText(img_rgb, str(count1),
                    (603, 610),
                    cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
        cv2.putText(img_rgb, str(round(calo1,2)),
                    (983, 610),
                    cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
        cv2.putText(img_rgb, str(count2),
                    (603, 740),
                    cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
        cv2.putText(img_rgb, str(round(calo2,2)),
                    (983, 740),
                    cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
        cv2.putText(img_rgb, str(count3),
                    (603, 870),
                    cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
        cv2.putText(img_rgb, str(round(calo3,2)),
                    (983, 870),
                    cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
        cv2.putText(img_rgb, str(count4),
                    (603, 1000),
                    cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
        cv2.putText(img_rgb, str(round(calo4,2)),
                    (983, 1000),
                    cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
        cv2.putText(img_rgb, str(calories),
                    (793, 1120),
                    cv2.FONT_HERSHEY_PLAIN, 5.5, (63, 66, 54), 7, cv2.LINE_AA)
        cv2.imshow('AI EXERCISE',img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()