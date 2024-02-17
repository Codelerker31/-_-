import threading
import cv2
import time
import datetime
from deepface import DeepFace


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

W, H = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

detection = False
face_match = True
detection_stopped_time = time.time()
vid1 = None
vid2 = None
fCounter = 0
dCounter = 0
img_num = 0
timer_start = False


DETECTION_WAIT_TIME = 5
fourcc = cv2.VideoWriter_fourcc(*"mp4v")


# This function detects faces in the frames
def detect_face(frame):
    global detection

    try:
        if DeepFace.extract_faces(frame, (W, H), detector_backend="opencv"):
            detection = True
        else:
            detection = False
    except ValueError:
        detection = False


# This function checks if detected face is recognized or not
def check_face(frame):
    global face_match, img_num

    # Cycle through images in Training Directory
    if face_match:
        img_path = f"Training Images/{img_num}.jpg"
        test_img = cv2.imread(img_path)
    elif ValueError:
        img_num = 0
        img_path = f"Training Images/{img_num}.jpg"
        test_img = cv2.imread(img_path)
    else:
        img_num += 1
        img_path = f"Training Images/{img_num}.jpg"
        test_img = cv2.imread(img_path)

    try:
        if DeepFace.verify(frame, test_img, model_name="SFace", detector_backend="opencv", enforce_detection=False)['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


while True:
    _, frame = cap.read()
    timer_count = time.time()

    # Runs the face detection concurrently after 30 iterations
    if dCounter % 30 == 0:
        try:
            threading.Thread(target=detect_face, args=(frame.copy(),)).start()
        except ValueError:
            pass
    dCounter += 1

    # If detection is true check if face is recognized
    if detection:
        current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

        # Runs the face recognition concurrently after 30 iterations
        if fCounter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        fCounter += 1

        # Record video of person stating if they are a User or Not
        if face_match and vid1 is None:
            vid1 = cv2.VideoWriter(f"{current_time}.mp4", fourcc, 30, (W, H))
        elif face_match is False and vid2 is None:
            vid2 = cv2.VideoWriter(f" UNIDENTIFIED PERSON! {current_time}.mp4", fourcc, 30, (W, H))
        elif face_match:
            cv2.putText(frame, "USER MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            vid1.write(frame)
        else:
            cv2.putText(frame, "UNIDENTIFIED PERSON!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            vid2.write(frame)

        detection_stopped_time = time.time()

    # Stop after 5 iterations of no detection
    else:
        if timer_count - detection_stopped_time >= DETECTION_WAIT_TIME:
            break

    cv2.imshow('Video', frame)

    # End program if q key is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break


print("Stopped Recording")
vid1.release()
vid2.release()
cap.release()
cv2.destroyAllWindows()
