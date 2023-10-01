import cv2
from deepface import DeepFace

# Загрузка модели для распознавания лиц
model = DeepFace.build_model("Facenet512")

# Загрузка каскадного классификатора для обнаружения лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Запуск видеопотока с веб-камеры
video_capture = cv2.VideoCapture('bb.mp4')

while True:
    # Чтение кадра из видеопотока
    ret, frame = video_capture.read()

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))

    # Идентификация и отображение результатов
    for (x, y, w, h) in faces:
        # Извлечение лица из кадра
        face = frame[y:y + h, x:x + w]

        # Предсказание идентичности лица
        predictions = DeepFace.find(img_path=face, db_path='faces', model_name="Facenet512", enforce_detection=False)[0].values.tolist()

        # Получение результата идентификации
        if predictions:
            label = predictions[0][0].split('\\')[1].split('/')[0]  # Костыль
        else:
            label = "???"

        # Отображение прямоугольника и метки на лице
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Отображение кадра с результатами
    cv2.imshow('Video', frame)

    # Прерывание цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video_capture.release()
cv2.destroyAllWindows()
