import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Model Loading
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Action labels (अपना डेटा डालें)
actions = ["hello", "thanks", "yes", "no", "iloveyou"]  # इसे अपने dataset के हिसाब से बदलें

# Mediapipe Hands Model
mp_hands = mp.solutions.hands

# Function to extract keypoints (Define this function)
def extract_keypoints(results):
    keypoints = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                keypoints.append(landmark.x)
                keypoints.append(landmark.y)
                keypoints.append(landmark.z)
    else:
        keypoints = [0] * 63  # Default keypoints if no hand is detected
    return keypoints

# Colors for visualization
colors = [(245, 117, 16)] * 20

def prob_viz(res, actions, input_frame, colors, threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Variables for prediction
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Define the region of interest
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)

        # Convert frame to RGB for mediapipe
        image = cv2.cvtColor(cropframe, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        # Extract keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        try:
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(round(res[np.argmax(res)] * 100, 2)) + "%")
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(round(res[np.argmax(res)] * 100, 2)) + "%")

                if len(sentence) > 1:
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]

        except Exception as e:
            print("Error:", e)

        # Display the output text
        cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
        cv2.putText(frame, "Output: " + ' '.join(sentence) + " " + ''.join(accuracy), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow('OpenCV Feed', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
