import tkinter as tk
from tkinter.ttk import Label
from tkinter.ttk import Button
from time import sleep
from utils import postprocess
import cv2 as cv
from tkinter import filedialog

root = tk.Tk()
root.geometry('900x600')
root.title('Helmet Detection App')
root.configure(background='LIGHTSKYBLUE2')

label1 = tk.Label(root, text='CHOOSE THE BELOW OPTIONS TO DETECT THE HELMET', font=('Verdana', 15), bg='LIGHTSKYBLUE2', fg='DARKSLATEGREY')
label1.place(relx=0.5, rely=0.3, anchor='center')

def start_camera():
    frame_count = 0
    frame_count_out = 0
    confThreshold = 0.5
    nmsThreshold = 0.4
    inpWidth = 416
    inpHeight = 416
    classesFile = "obj.names"

    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    modelConfiguration = "yolov3-obj.cfg";
    modelWeights = "yolov3-obj_2400.weights";

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    layersNames = net.getLayerNames()
    output_layer = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        net.setInput(blob)

        outs = net.forward(output_layer)

        postprocess(frame, outs, confThreshold, nmsThreshold, classes)
        cv.imshow('img', frame)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    cap.release()
    cv.destroyAllWindows()

def upload_file():
    file_path = filedialog.askopenfilename()
    cap = cv.VideoCapture(file_path)

    frame_count = 0
    frame_count_out = 0
    confThreshold = 0.5
    nmsThreshold = 0.4
    inpWidth = 416
    inpHeight = 416
    classesFile = "obj.names"

    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    modelConfiguration = "yolov3-obj.cfg";
    modelWeights = "yolov3-obj_2400.weights";

    net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    layersNames = net.getLayerNames()
    output_layer = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

        net.setInput(blob)

        outs = net.forward(output_layer)

        postprocess(frame, outs, confThreshold, nmsThreshold, classes)
        cv.imshow('img', frame)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

button1 = tk.Button(root, text='START CAMERA', font=('Verdana', 14), command=start_camera, bg='dodger blue', fg='white')
button1.place(relx=0.4, rely=0.6, anchor='center')

button2 = tk.Button(root, text='UPLOAD FILE', font=('Verdana', 14), command=upload_file, bg='dodger blue', fg='white')
button2.place(relx=0.6, rely=0.6, anchor='center')
root.configure(background='black')
root.mainloop()
