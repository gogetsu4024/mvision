from flask import Response, request, jsonify, send_file, send_from_directory, app
from flask_restful import Resource
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os


ALLOWED_EXTENSIONS = set(['mp4', 'avi'])
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class AllRequests(Resource):
	def post(self):
		if 'file' not in request.files:
			resp = jsonify({'message': 'No file part in the request'})
			resp.status_code = 400
			return resp
		file = request.files['file']
		if file.filename == '':
			resp = jsonify({'message': 'No file selected for uploading'})
			resp.status_code = 400
			return resp
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join('./', filename))

			model = load_model('model-018.model')
			face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

			cap = cv2.VideoCapture('something.mp4')
			cap.set(3, 640)
			cap.set(4, 480)

			fourcc = cv2.VideoWriter_fourcc(*'mp4v')

			if (cap.isOpened() == False):
				print("Unable to read camera feed")
			frame_width = int(cap.get(3))
			frame_height = int(cap.get(4))

			out = cv2.VideoWriter('outpy.mp4', fourcc, 10, (frame_width, frame_height))

			labels_dict = {0: 'MASK', 1: 'NO MASK'}
			color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
			while (cap.read()):

				ret, frame = cap.read()
				if (ret):
					gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				else:
					break

				faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

				for (x, y, w, h) in faces:

					face_img = gray[y:y + w, x:x + w]
					resized = cv2.resize(face_img, (100, 100))
					normalized = resized / 255.0
					reshaped = np.reshape(normalized, (1, 100, 100, 1))
					result = model.predict(reshaped)

					label = np.argmax(result, axis=1)[0]

					cv2.rectangle(frame, (x, y), (x + w, y + h), color_dict[label], 4)
					cv2.rectangle(frame, (x, y - 40), (x + w, y), color_dict[label], 4)
					cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_ITALIC, 1, (255, 255, 255), 4)

					if (labels_dict[label] == 'Mask'):
						print("No Beep")
					elif (labels_dict[label] == 'NoMask'):

						print("Beep")

				out.write(frame)
				cv2.imshow('Mask Detection App', frame)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			cap.release()
			out.release()
			cv2.destroyAllWindows()
			return send_from_directory("", "outpy.mp4", as_attachment=True)
		else:
			resp = jsonify({'message': 'Allowed file types are mp4,avi'})
			resp.status_code = 400
			return resp


