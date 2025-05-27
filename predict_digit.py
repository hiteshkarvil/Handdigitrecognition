import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model.h5")

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype("float32") / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    processed = preprocess(roi)
    prediction = model.predict(processed)
    digit = np.argmax(prediction)

    cv2.putText(frame, f'Prediction: {digit}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)
    cv2.imshow("Handwritten Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
