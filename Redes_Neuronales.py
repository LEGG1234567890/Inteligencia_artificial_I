import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' 
import cv2
import numpy as np
import tensorflow as tf
model=tf.keras.models.load_model("my_model.keras")

def procesar_frame(frame, roi_x=100, roi_y=100, roi_width=200, roi_height=200):
    roi=frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]
    gray=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray=255-gray
    resized=cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    img=resized.astype('float32')/255
    input_img=img.reshape(1, 28, 28, 1)
    prediction=model.predict(input_img, verbose=0)
    predicted_label=np.argmax(prediction)
    confidence=prediction[0][predicted_label]*100
    return predicted_label, confidence, resized

def main():
    cap =cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam")
        return
    roi_x, roi_y=100, 100
    roi_width, roi_height=200, 200
    while True:
        ret, frame=cap.read()
        if not ret:
            print("Error: No se pudo capturar el frame")
            break
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_width, roi_y+roi_height), (0, 255, 0), 2)
        predicted_label, confidence, processed_img=procesar_frame(frame, roi_x, roi_y, roi_width, roi_height)
        cv2.putText(frame, f"Prediccion: {predicted_label} ({confidence:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Reconocimiento de Digitos en Tiempo Real', frame)
        cv2.imshow('Imagen Procesada (28x28)', cv2.resize(processed_img, (140, 140), interpolation=cv2.INTER_NEAREST))
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()