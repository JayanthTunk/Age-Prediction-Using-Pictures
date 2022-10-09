import numpy as np
import cv2
import dlib


img = cv2.imread('/home/jay/Documents/AD P/Input/MB.jpg')
img = cv2.resize(img, (720, 640))
frame = img.copy()

a_w = "/home/jay//Documents/AD P/age.prototxt"
a_c = "/home/jay//Documents/AD P/age.caffemodel"
age_Net = cv2.dnn.readNet(a_c, a_w)

ageList = ['(0-3)', '(3-7)', '(7-14)', '(14-22)', '(22-35)', '(35-44)', '(44-57)', '(57-100)']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

fH = img.shape[0]
fW = img.shape[1]

Boxes = []
msg = 'Approximate Age' 
face_detector = dlib.get_frontal_face_detector()
img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = face_detector(img_gray)
if not faces:
	msg = 'No face detected'
	cv2.putText(img, f'{msg}', (40, 40),
				cv2.FONT_HERSHEY_SIMPLEX, 2, (200), 2)
	cv2.imshow('Age detected', img)
	cv2.waitKey(0)

else:
	for face in faces:
		x = face.left() 
		y = face.top()
		x2 = face.right()
		y2 = face.bottom()

		box = [x, y, x2, y2]
		Boxes.append(box)
		cv2.rectangle(frame, (x, y), (x2, y2),
					(00, 200, 200), 2)

	for box in Boxes:
		face = frame[box[1]:box[3], box[0]:box[2]]

		blob = cv2.dnn.blobFromImage(
			face, 1.0, (227, 227), model_mean, swapRB=False)

		age_Net.setInput(blob)
		age_preds = age_Net.forward()
		age = ageList[age_preds[0].argmax()]

		cv2.putText(frame, f'{msg}:{age}', (box[0],
											box[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8,
					(0, 255, 255), 2, cv2.LINE_AA)

		cv2.imshow("Age Detection by Jayanth Tunk", frame)
		cv2.waitKey(0)
