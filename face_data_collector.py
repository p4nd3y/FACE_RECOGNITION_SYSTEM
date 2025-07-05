import cv2
import numpy as np
import os


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

dataset_path = "./face_dataset/"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

file_name = input("Enter the name of the person: ").strip().lower()

face_data = []
skip = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    faces = sorted(faces, key=lambda f: f[2]*f[3])  

    # Pick the largest face
    for (x, y, w, h) in faces[-1:]:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # Extract (Crop out the face region)
        offset = 10
        face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        if skip % 10 == 0:
            face_data.append(face_section)
            print(f"[INFO] Saved frame {len(face_data)}")

        skip += 1

    cv2.imshow("Collecting Face Data", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or len(face_data) >= 100:
        break

# Convert face list into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))

# Save as .npy file
np.save(dataset_path + file_name + ".npy", face_data)
print(f"[SUCCESS] Data saved at: {dataset_path + file_name + '.npy'}")

cap.release()
cv2.destroyAllWindows()
