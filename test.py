#test for camera

import cv2 as cv

camera = cv.VideoCapture(0, cv.CAP_DSHOW)

# Проверяем, открыта ли камера
if not camera.isOpened():
    print("Не удалось открыть камеру")
    exit()

# Бесконечный цикл захвата кадров
while True:
    # Считываем кадр
    ret, frame = camera.read()

    if not ret:
        print("Не удалось считать кадр")
        break

    # Отображаем кадр
    cv.imshow('Видеопоток с камеры', frame)

    # Выход из цикла по нажатию 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
camera.release()
cv.destroyAllWindows()