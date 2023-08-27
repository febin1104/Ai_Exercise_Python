import cv2,os,urllib.request
import numpy as np
import mediapipe as mp
from django.conf import settings

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

class VideoCamera(object):
	def __init__(self):
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_pose = mp.solutions.pose
		self.counter = 0
		self.count1 = 0
		self.count2 = 0
		self.count3 = 0
		self.count4 = 0
		self.stage = None
		self.exercise = "AI EXERCISE"
		self.calories = 0
		self.calo1 = 0
		self.calo2 = 0
		self.calo3 = 0
		self.calo4 = 0
		self.angle1=None
		self.angle2 = None
		self.angle3 = None
		self.cap = cv2.VideoCapture(0)
		self.cap.set(3, 1280)
		self.cap.set(4, 720)

	def reset(self):
		self.stage=None
		self.counter=0

	def __del__(self):
		self.cap.release()

	def intro(self):
		path = r"D:\FEBIN RAJAN\pythonProject\Exercise\resources\AI_EXERCISE1.jpg"
		img_rgb = cv2.imread(path, 1)
		ret, jpeg = cv2.imencode('.jpg', img_rgb)
		return jpeg.tobytes()

	def pushup(self):
		self.exercise = "PUSH UP"
		## Setup mediapipe instance
		with self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
			while self.cap.isOpened():
				ret, frame = self.cap.read()

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
					shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
								landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
					elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
							 landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
					wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
							 landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

					# Calculate angle
					self.angle1 = calculate_angle(shoulder, elbow, wrist)

					cv2.putText(image, str(self.angle1),
								tuple(np.multiply(elbow, [1280, 720]).astype(int)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
								)

					# Curl counter logic
					if self.angle1 > 160:
						self.stage = "up"
					if self.angle1 < 70 and self.stage == 'up':
						self.stage = "down"
						self.counter += 1
						self.count1 += 1
						self.calo1 += 0.36
						self.calories += 0.36
						self.calories = round(self.calories, 2)
						#print(self.counter)

				except:
					pass
				# Exercise Data
				cv2.putText(image, 'Exercise: ', (5, 25), cv2.FONT_HERSHEY_PLAIN, 1.5,(0, 255, 0), 2)
				cv2.putText(image, self.exercise,(125, 25),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
				# Counter Data
				cv2.putText(image, 'Counter: ', (5, 45), cv2.FONT_HERSHEY_PLAIN, 1.5,(0, 255, 0), 2)
				cv2.putText(image, str(self.counter),(125, 45),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
				# Stage Data
				cv2.putText(image, 'Stage: ', (5, 65), cv2.FONT_HERSHEY_PLAIN, 1.5,(0, 255, 0), 2)
				cv2.putText(image, self.stage,(125, 65),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
				# Calories Data
				cv2.putText(image, 'Calories: ', (5, 85), cv2.FONT_HERSHEY_PLAIN, 1.5,(0, 255, 0), 2)
				cv2.putText(image, str(self.calories),(125, 85),cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

				# Render detections
				self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
												self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
																			circle_radius=2),
												self.mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2,
																			circle_radius=2)
												)
				ret, jpeg = cv2.imencode('.jpg', image)
				return jpeg.tobytes()

	def squats(self):
		self.exercise = "Squats"
		## Setup mediapipe instance
		with self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
			while self.cap.isOpened():
				ret, frame = self.cap.read()

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
					hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
						   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
					knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
							landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
					ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
							 landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

					# Calculate angle
					self.angle2 = calculate_angle(hip, knee, ankle)

					cv2.putText(image, str(self.angle2),
								tuple(np.multiply(knee, [1280, 720]).astype(int)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
								)

					# Curl counter logic
					if self.angle2 > 160:
						self.stage = "up"
					if self.angle2 < 70 and self.stage == 'up':
						self.stage = "down"
						self.counter += 1
						self.count2 += 1
						self.calo2 += 0.32
						self.calories += 0.32
						self.calories = round(self.calories, 2)
						#print(self.counter)

				except:
					pass
				# Exercise Data
				cv2.putText(image, 'Exercise: ', (5, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				cv2.putText(image, self.exercise, (125, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2,cv2.LINE_AA)
				# Counter Data
				cv2.putText(image, 'Counter: ', (5, 45), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0),2)
				cv2.putText(image, str(self.counter), (125, 45), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2,cv2.LINE_AA)
				# Stage Data
				cv2.putText(image, 'Stage: ', (5, 65), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				cv2.putText(image, self.stage, (125, 65), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
				# Calories Data
				cv2.putText(image, 'Calories: ', (5, 85), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				cv2.putText(image, str(self.calories), (125, 85), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2,cv2.LINE_AA)

				# Render detections
				self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
											  self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
																	 circle_radius=2),
											  self.mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2, circle_radius=2)
											  )

				ret, jpeg = cv2.imencode('.jpg', image)
				return jpeg.tobytes()

	def situp(self):
		self.exercise = "SIT UP"
		## Setup mediapipe instance
		with self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
			while self.cap.isOpened():
				ret, frame = self.cap.read()

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
					shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
								landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
					hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
						   landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
					knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
							landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]

					# Calculate angle
					self.angle3 = calculate_angle(shoulder, hip, knee)

					cv2.putText(image, str(self.angle3),
								tuple(np.multiply(hip, [1280, 720]).astype(int)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
								)

					# Curl counter logic
					if self.angle3 > 105:
						self.stage = "down"
					if self.angle3 < 55 and self.stage == 'down':
						self.stage = "up"
						self.counter += 1
						self.count3 += 1
						self.calo3 += 0.25
						self.calories += 0.25
						self.calories = round(self.calories, 2)
						#print(self.counter)
				except:
					pass
				# Exercise Data
				cv2.putText(image, 'Exercise: ', (5, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				cv2.putText(image, self.exercise, (125, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2,cv2.LINE_AA)
				# Counter Data
				cv2.putText(image, 'Counter: ', (5, 45), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				cv2.putText(image, str(self.counter), (125, 45), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2,cv2.LINE_AA)
				# Stage Data
				cv2.putText(image, 'Stage: ', (5, 65), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				cv2.putText(image, self.stage, (125, 65), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
				# Calories Data
				cv2.putText(image, 'Calories: ', (5, 85), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				cv2.putText(image, str(self.calories), (125, 85), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2,cv2.LINE_AA)

				# Render detections
				self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
											  self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
																	 circle_radius=2),
											  self.mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2, circle_radius=2)
											  )
				ret, jpeg = cv2.imencode('.jpg', image)
				return jpeg.tobytes()

	def biceps(self):
		self.exercise = "BICEP CURL"
		## Setup mediapipe instance
		with self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
			while self.cap.isOpened():
				ret, frame = self.cap.read()

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
					shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
								landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
					elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
							 landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
					wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
							 landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

					# Calculate angle
					self.angle4 = calculate_angle(shoulder, elbow, wrist)

					cv2.putText(image, str(self.angle4),
								tuple(np.multiply(elbow, [1280, 720]).astype(int)),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
								)

					# Curl counter logic
					if self.angle4 > 160:
						self.stage = "down"
					if self.angle4 < 30 and self.stage == 'down':
						self.stage = "up"
						self.counter += 1
						self.count4 += 1
						self.calo4 += 0.9
						self.calories += 0.9
						self.calories = round(self.calories, 2)
						#print(self.counter)

				except:
					pass
				# Exercise Data
				cv2.putText(image, 'Exercise: ', (5, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				cv2.putText(image, self.exercise, (125, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2,cv2.LINE_AA)
				# Counter Data
				cv2.putText(image, 'Counter: ', (5, 45), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				cv2.putText(image, str(self.counter), (125, 45), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2,cv2.LINE_AA)
				# Stage Data
				cv2.putText(image, 'Stage: ', (5, 65), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				cv2.putText(image, self.stage, (125, 65), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
				# Calories Data
				cv2.putText(image, 'Calories: ', (5, 85), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
				cv2.putText(image, str(self.calories), (125, 85), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2,cv2.LINE_AA)

				# Render detections
				self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
											  self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
																	 circle_radius=2),
											  self.mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2, circle_radius=2)
											  )

				ret, jpeg = cv2.imencode('.jpg', image)
				return jpeg.tobytes()

	def result(self):
		path = r"D:\FEBIN RAJAN\pythonProject\Exercise\resources\EXERCISE_CHART1.jpg"
		img_rgb = cv2.imread(path, 1)
		cv2.putText(img_rgb, str(self.count1),(603, 610),cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
		cv2.putText(img_rgb, str(round(self.calo1, 2)),(983, 610),cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
		cv2.putText(img_rgb, str(self.count2),(603, 740),cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
		cv2.putText(img_rgb, str(round(self.calo2, 2)),(983, 740),cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
		cv2.putText(img_rgb, str(self.count3),(603, 870),cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
		cv2.putText(img_rgb, str(round(self.calo3, 2)),(983, 870),cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
		cv2.putText(img_rgb, str(self.count4),(603, 1000),cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
		cv2.putText(img_rgb, str(round(self.calo4, 2)),(983, 1000),cv2.FONT_HERSHEY_PLAIN, 4, (63, 66, 54), 5, cv2.LINE_AA)
		cv2.putText(img_rgb, str(self.calories),(793, 1120),cv2.FONT_HERSHEY_PLAIN, 5.5, (63, 66, 54), 7, cv2.LINE_AA)

		ret, jpeg = cv2.imencode('.jpg', img_rgb)
		return jpeg.tobytes()