#  Обнаружение человека с помощью камеры видеонаблюдения

Проект написан на opensource фреймворке [Streamlit](https://streamlit.io/) и модели Yolo8 от [Ultralytics](https://www.ultralytics.com/ru/). \
Деплой на серверах Streamlit.\
Million thanks to Streamlit and Ultralytics. :heart: 🚀

---
**Проблема**: охранная система, построенная на датчиках движения, срабатывает на кошку, камера видеонаблюдения работает отдельно и только снимает происходящее.

**Решение**: объединить камеру и охранную систему, чтобы при обнаружении человека, сигнал поступал на датчик.

---

Для демонстрации работы выбраны низкокачественные демо-видео с камеры ночного видения и домашних камер низкого разрешения, ютуб ролики также отбираются в качестве 720.

---
_Это часть проекта, для демонстрации работы детекции с моделью Yolo8._
### Проект когда-нибудь будет доступен по адресу [ссылка]()
Проект работает в 4х режимах:
- Video: загруженные в проект демо видео.
- RTSP: видео с камеры наблюдения.
- Youtube: видео по ссылке с ютуб.
- MyVideo: загруженное пользователем видео

---
### Примеры работы проекта:
**Детекция с камеры ночного видения с человеком:**
![image](https://github.com/JuliaBars/thief_detection_with_cctv_cam/assets/107411145/8e9d12cd-a73f-4e54-b064-8f27f9058e54)
**Видео без человека:**
![image](https://github.com/JuliaBars/thief_detection_with_cctv_cam/assets/107411145/18b5d732-4a4b-43eb-b71e-7b43ed614d54)
**Детекция потокового видео с ютуб:**
![image](https://github.com/JuliaBars/thief_detection_with_cctv_cam/assets/107411145/19b9edd7-c767-4869-a862-c2279c0387b2)


### Локальный запуск:
```
git clone https://github.com/JuliaBars/thief_detection_with_cctv_cam
cd thief_detection_with_cctv_cam
pip install -r requirements.txt
```
Запустите тесты:
```
pytest
```
И запустите проект:
```
streamlit run app.py
```

_For "backbone" many thanks to [CodingMantras](https://github.com/CodingMantras/yolov8-streamlit-detection-tracking/)_
