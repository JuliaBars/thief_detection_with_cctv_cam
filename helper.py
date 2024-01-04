import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy
import streamlit as st
from loguru import logger
from pytube import YouTube
from ultralytics import YOLO

import settings

logger.add('file_1.log', rotation='50 MB')


@st.cache_resource
def load_model(model_path: Path) -> YOLO:
    """ Загрузка Yolo модели.
    Args:
    - model_path: путь до скачанной модели"""
    model = YOLO(model_path)
    return model


def _display_detected_frames(
        conf: float,
        model: YOLO,
        st_frame: st.delta_generator.DeltaGenerator,
        image: numpy.ndarray) -> bool:
    """
    Отображение боксов и класса в кадре.
    Args:
        conf: уверенность модели (устанавливается пользователем)
        model: модель yolo8
        st_frame: область для отрисовки
        image: кадр для детекции
    Return:
        1: человек обнаружен
        0: не обнаружен
    """
    image = cv2.resize(image, (720, int(720*(9/16))))
    res = model.predict(image, conf=conf)
    res_plotted = res[0].plot()
    human_inside = False
    # YOLO8: у человека класс 0
    if 0 in res[0].boxes.cls:
        human_inside = True
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    return human_inside


def play_youtube_video(conf: float, model: YOLO) -> None:
    """
    Обработка видео по ссылки youtube.
    Args:
        conf: уверенность модели (устанавливается пользователем)
        model: модель yolo8
    """
    source_youtube = st.sidebar.text_input("Ссылка на YouTube")

    if st.sidebar.button('Detect'):
        try:
            yt = YouTube(source_youtube)
            stream: Any = yt.streams.filter(
                file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)
            st_frame = st.empty()
            warning_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    human_inside = _display_detected_frames(conf,
                                                            model,
                                                            st_frame,
                                                            image,
                                                            )
                    if human_inside:
                        logger.warning('Обнаружен человек')
                        warning_frame.warning('Обнаружен человек', icon="⚠️")
                else:
                    vid_cap.release()
                    break
            if not human_inside:
                st.success('На видео не обнаружено людей', icon="✅")
        except Exception as e:
            st.sidebar.error('Error loading video: ' + str(e))
            logger.error('Ошибка при загрузке: ', str(e))


def play_rtsp_stream(conf: float, model: YOLO) -> None:
    """
    Обработка потокового видео с RTSP камеры.
    Args:
        conf: уверенность модели (устанавливается пользователем)
        model: модель yolo8
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    if st.sidebar.button('Detect'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            warning_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    human_inside = _display_detected_frames(conf,
                                                            model,
                                                            st_frame,
                                                            image,
                                                            )
                    if human_inside:
                        logger.warning('Обнаружен человек')
                        warning_frame.warning('Обнаружен человек', icon="⚠️")
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))
            logger.error('Ошибка при загрузке: ', str(e))


def play_stored_video(conf: float, model: YOLO) -> None:
    """
    Обработка демовидео, предзагруженных в проект.
    Args:
        conf: уверенность модели (устанавливается пользователем)
        model: модель yolo8
    """
    source_vid: Any = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    if st.sidebar.button('Detect'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            warning_frame = st.empty()
            while vid_cap.isOpened():
                success, image = vid_cap.read()
                if success:
                    human_inside = _display_detected_frames(conf,
                                                            model,
                                                            st_frame,
                                                            image,
                                                            )

                    if human_inside:
                        logger.warning('Обнаружен человек')
                        warning_frame.warning('Обнаружен человек', icon="⚠️")
                else:
                    vid_cap.release()
                    break
            if not human_inside:
                st.success('На видео не обнаружено людей', icon="✅")
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
            logger.error('Ошибка при загрузке: ', str(e))


def play_loaded_video(conf: float, model: YOLO) -> None:
    """
    Обработка видео, загруженных пользователем.
    Args:
        conf: уверенность модели (устанавливается пользователем)
        model: модель yolo8
    """
    uploaded_file: Any = st.sidebar.file_uploader(
        "Choose a video (max 200Mb)...")

    if st.sidebar.button('Detect'):
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tfile:
                tfile.write(uploaded_file.read())
                vid_cap = cv2.VideoCapture(tfile.name)
                st_frame = st.empty()
                warning_frame = st.empty()
                while (vid_cap.isOpened()):
                    success, image = vid_cap.read()
                    if success:
                        human_inside = _display_detected_frames(conf,
                                                                model,
                                                                st_frame,
                                                                image,
                                                                )
                        if human_inside:
                            logger.warning('Обнаружен человек')
                            warning_frame.warning(
                                'Обнаружен человек', icon="⚠️")
                    else:
                        vid_cap.release()
                        break
                if not human_inside:
                    st.success('На видео не обнаружено людей', icon="✅")
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
            logger.error('Ошибка при загрузке: ', str(e))
