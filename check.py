import cv2
import argparse
import tensorflow as tf
import datetime
from imutils.video import VideoStream
from utils import detector_utils as detector_utils
from datetime import date
import numpy as np


lst1=[]
lst2=[]
TRAINED_MODEL_DIR = 'trained_model'
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/frozen_inference_graph.pb'
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/label_map.pbtxt'

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())
def detect_objects(image_np, detection_graph, sess):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)

def load_inference_graph():


    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess

detection_graph, sess = load_inference_graph()
def drawsafelines(image_np,Line_Perc2):

    posii=int(image_np.shape[1]-(image_np.shape[1]/3))
    cv2.putText(image_np,'RED_Line : Fruit Sorting Side',
                        (posii,50),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,0,255), 1, cv2.LINE_AA)

    Line_Position2 = int(image_np.shape[1] * (Line_Perc2 / 100))


    cv2.line(img=image_np, pt1=(Line_Position2, 0), pt2=(Line_Position2, image_np.shape[0]), color=(255, 0, 0),
             thickness=2, lineType=8, shift=0)

    return Line_Position2;
if __name__ == '__main__':
    score_thresh = 0.80

    vs = VideoStream(1).start()  #when use laptop webcam put 0 insted of 1

    Line_Perc2 = float(80)

    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None, None)
    cv2.namedWindow('Fruit Sorting Project', cv2.WINDOW_NORMAL)



    try:
        while True:
            frame = vs.read()
            frame = np.array(frame)

            if im_height == None:
                im_height, im_width = frame.shape[:2]

            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")
            boxes, scores, classes = detect_objects(frame, detection_graph, sess)


            Line_Position2 = drawsafelines(frame, Line_Perc2)
            detector_utils.draw_box_on_image(
                 score_thresh, scores, boxes, classes, im_width, im_height, frame, Line_Position2)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:

                detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Fruit Sorting Project', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.stop()
                    break



        print("Average FPS: ", str("{0:.2f}".format(fps)))

    except KeyboardInterrupt:
        today = date.today()
        print("Average FPS: ", str("{0:.2f}".format(fps)))