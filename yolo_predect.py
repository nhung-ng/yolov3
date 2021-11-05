
from utils import *

def startModel():
    labelsPath="data/coco.names"
    cfgpath="cfg/yolov3.cfg"
    wpath="weights/yolov3.weights"
    Lables=get_labels(labelsPath)
    CFG=get_config(cfgpath)
    Weights=get_weights(wpath)
    nets=load_model(CFG,Weights)
    Colors=get_colors(Lables)
    return nets,Lables,Colors

def predection(image,net,LABELS,COLORS):
    (H, W) = image.shape[:2]

    # get output layer names  YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] # return index of ouputlayers

    # convert image into blob for input
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),(0,0,0),
                                 swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()

    layerOutputs = net.forward(ln)
    print(layerOutputs)
    
    end = time.time()
    # outputs can give bbox, probs

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # init lists of detected bbox, confidences, and # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    _time_start = time.time() #đo thời gian thực thi của model

    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # class ID and confidence (i.e., probability) of current object
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            #filter meaning prob is greater than the minimum prob
            if confidence > confthres:
                #center (x, y)-coord of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # top & left corner of the bbox
                x = int(round(centerX - width / 2.0))
                y = int(round(centerY - height / 2.0))

                # update list of bounding box , confidences,# and class IDs
                boxes.append([x, y, int(round(width)), int(round(height))])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-max_suppression to suppress weak, overlapping bbox
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres, nmsthres)

    # exists least one detection
    if len(idxs) > 0:
        # loop over the indexes are keeping
        for i in idxs.flatten():
            # extract the bounding box coord
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bbox rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color,2)
            text = "{}: {:.2f}".format(LABELS[classIDs[i]], round(confidences[i]*100,2))
            cv2.putText(image, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX ,0.5, color, 2) 
            #cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_PLAIN ,0.7, (0,0,0), 1)
    _time_end = time.time()
    print("YOLO Execution time: " + str(_time_end - _time_start))
    return image



