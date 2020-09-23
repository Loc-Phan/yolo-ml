from flask import *
import cv2
import numpy as np
import glob
import random
import os
from recognition import E2E
from pathlib import Path
import argparse
import time

app = Flask(__name__, template_folder='template')


@app.route('/')
def upload():
    if (os.path.isfile("./static/savedImage.jpg")):
        os.remove("./static/savedImage.jpg")
    return render_template('index.html')


@app.route('/success', methods=['POST', 'GET'])
def success():
    if request.method == 'POST':
        countImageName = random.randint(1, 999)
        t = str(countImageName)
        select = request.form.get("selected_id")
        f = request.files['file']

        f.save("./static/"+str(countImageName)+f.filename)
        # Insert here the path of your images
        img_path = "./static/" + str(countImageName) + f.filename
        # Loading image
        img = cv2.imread(img_path)
        if select == "3":
            start = time.time()
            image = model.predict(img)
            ff = 'savedImage' + f.filename + select + t + '.jpg'
            cv2.imwrite("./static/" + ff, image)
            end = time.time()
            return render_template("success.html", name=ff, time=str(end-start), names=str(countImageName)+f.filename)
        img = cv2.resize(img, None, fx=1, fy=1)
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(
            img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        if select == "2":
            net_.setInput(blob)
            outs = net_.forward(output_layers_)
        else:
            net.setInput(blob)
            outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        start = time.time()
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                if select == "2":
                    label = str(classes_[class_ids[i]])
                    color = colors_[class_ids[i]]
                else:
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

        ff = 'savedImage'+f.filename+select+str(countImageName)+'.jpg'
        cv2.imwrite("./static/" + ff, img)
        end = time.time()
        return render_template("success.html", name=ff, time=str(end-start), names=str(countImageName)+f.filename)


if __name__ == '__main__':
    net = cv2.dnn.readNet("./weights/yolov3.weights", "./cfg/yolov3_.cfg")
    net_ = cv2.dnn.readNet("./weights/yolov3.backup", "./cfg/yolov3.cfg")
    # Name custom object
    classes = open("./cfg/coco.names").read().strip().split("\n")
    classes_ = open("./cfg/yolo.names").read().strip().split("\n")
    layer_names = net.getLayerNames()
    layer_names_ = net_.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    output_layers_ = [layer_names_[i[0] - 1]
                      for i in net_.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    colors_ = np.random.uniform(0, 255, size=(len(classes_), 3))
    # tenbien
    model = E2E()
    app.run(debug=True, port=7000)
