from pathlib import Path

import streamlit as st
from loguru import logger
from ultralytics import YOLO

import helper
import settings

st.set_page_config(
    page_title="Обнаружение человека для охранной сигнализации",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': """Video: загруженные в проект демо видео.
        RTSP - видео с камеры наблюдения.
        Youtube - видео по ссылке с ютуб.
        MyVideo - загруженное пользователем видео"""
    }
)

st.header('Обнаружение человека для охранной сигнализации', divider='rainbow')
st.subheader('Выберите источник видео и нажмите Detect')
st.caption('''Результат работы под видео.
           Подробнее о режимах работы в About (право-верх)''')


st.sidebar.header('Настройки')
confidence: float = float(st.sidebar.slider(
    'Выберите уверенность модели', 25, 100, 75)) / 100

model_path: Path = Path(settings.DETECTION_MODEL)


try:
    model: YOLO = helper.load_model(model_path)
except Exception as ex:
    st.error(f'Не удалось загрузить модель: {model_path}')
    st.error(ex)
    logger.error(f'Не удалось загрузить модель: {model_path}')
    logger.error('Ошибка при загрузке модели: ', str(ex))

source_radio = st.sidebar.radio(
    'Выберите источник видео', settings.SOURCES_LIST)

if source_radio == settings.DEMOVIDEO:
    helper.play_stored_video(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

elif source_radio == settings.MYVIDEO:
    helper.play_loaded_video(confidence, model)

else:
    st.error('Выберите корректный тип ресурса')
    logger.error('Ошибка с ресурсом')
