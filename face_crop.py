import cv2
import age_number

# Код по распознаванию лиц взят с этого сайта http://chel-center.ru/python-yfc/2020/02/22/opencv-shpargalka/#faces
# Путь к модели
FACE_DETECTOR_PATH = r'models/haarcascade_frontalface_default.xml'


def crop_face(img, scale_factor=1.1, min_neighbors=19, face_detector_path=FACE_DETECTOR_PATH):
    # Загружаем модель
    face_cascade = cv2.CascadeClassifier(face_detector_path)
    # Переводим модель в серые цвета
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Распознаем лица
    # Параметр scaleFactor. Некоторые лица могут быть больше других, поскольку находятся ближе,
    # чем остальные. Этот параметр компенсирует перспективу.
    # Алгоритм распознавания использует скользящее окно во время распознавания объектов.
    # Параметр minNeighbors определяет количество объектов вокруг лица.
    # То есть чем больше значение этого параметра, тем больше аналогичных объектов необходимо
    # алгоритму, чтобы он определил текущий объект, как лицо. Слишком маленькое значение
    # увеличит количество ложных срабатываний, а слишком большое сделает алгоритм более требовательным.
    # minSize — непосредственно размер этих областей.
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    for (x, y, w, h) in faces:
        # Увеличим площадь прямоугольника вокруг лиц.
        # Значения могут быть только целочиленными.
        # Если значения по х или у меньше нуля, берем 0
        # альфа - кэффициент увеличения прямоугольника
        alpha = 1.5
        w_for_model = int(w * alpha)
        x_for_model = int(max(x - (w_for_model - w) / 2, 0))
        h_for_model = int(h * alpha)
        y_for_model = int(max(y - (h_for_model - h) / 2, 0))
        # Чтобы в модель передавался квадрат, а не прямоугольник
        w_for_model = h_for_model
        # Обрезаем исходную картинку прямоугольником (точнее квадратом)
        img_for_model = img[y_for_model:y_for_model + h_for_model,
                            x_for_model:x_for_model + w_for_model]
        age_num = age_number.age_recognition_func(img_for_model)

        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        x_for_text, y_for_text = int(x + (w / 2) - 20), y - 10
        img = cv2.putText(img, str(age_num), (x_for_text, y_for_text),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.5, (150, 150, 0), 3)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, len(faces)
