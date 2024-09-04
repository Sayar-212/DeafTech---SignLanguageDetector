import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Load the trained model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Initialize text-to-speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Update the labels dictionary as per your classes
labels_dict = {str(i): str(i) for i in range(10)}  # 0-9
labels_dict.update({chr(i): chr(i) for i in range(65, 91)})  # A-Z

recognized_text = ""

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data_aux = []
            x_, y_ = [], []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # Predict the character
            prediction = model.predict([np.asarray(data_aux)])
            predicted_label = prediction[0]  # Predicted label should match one in labels_dict

            # Debugging output
            print(f"Predicted label: {predicted_label}")
            print(f"Available labels: {labels_dict.keys()}")

            # Check if the predicted label is in the dictionary
            if predicted_label in labels_dict:
                predicted_character = labels_dict[predicted_label]

                # Append the predicted character to the recognized text
                recognized_text += predicted_character

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, recognized_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                print(f"Warning: Predicted label {predicted_label} not found in labels_dict")

    cv2.imshow('Sign Language Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Speak the recognized text
if recognized_text:
    print(f'Recognized text: {recognized_text}')
    engine.say(recognized_text)
    engine.runAndWait()

cap.release()
cv2.destroyAllWindows()
