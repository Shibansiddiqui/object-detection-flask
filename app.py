from flask import Flask, render_template, Response
import cv2
import os

app = Flask(__name__)

thres = 0.45  # Threshold to detect objects
classNames = []

# Load class names
script_dir = os.path.dirname(os.path.realpath(__file__))
classFile = os.path.join(script_dir, 'coco.names')
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load the detection model
ssd_path = os.path.join(script_dir, 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
frozen_path = os.path.join(script_dir, 'frozen_inference_graph.pb')
net = cv2.dnn_DetectionModel(frozen_path, ssd_path)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Function to process frames
def process_frame(frame):
    classIds, confs, bbox = net.detect(frame, confThreshold=thres)

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if 0 <= classId - 1 < len(classNames):
                class_name = classNames[classId - 1].upper()
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, class_name, (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return frame

# Camera capture function
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)