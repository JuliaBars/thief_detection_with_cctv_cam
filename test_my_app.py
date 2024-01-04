from os import listdir
from pathlib import Path

import cv2
import numpy
import pytest
import streamlit as st

import settings
from helper import YOLO, YouTube, _display_detected_frames, load_model


@pytest.fixture
def model():
    model_path = Path(settings.DETECTION_MODEL)
    return YOLO(model_path)


def test_load_model():
    model_path = Path(settings.DETECTION_MODEL)
    model = load_model(model_path)
    assert isinstance(model, YOLO)


def test_no_human_in_frame(model):
    conf = 0.5
    st_frame = st.empty()
    image = numpy.zeros((720, 1280, 3), dtype=numpy.uint8)
    human_inside = _display_detected_frames(conf, model, st_frame, image)
    assert not human_inside


def test_human_in_frame(model):
    conf = 0.8
    st_frame = st.empty()
    image = cv2.imread(settings.TEST_PIC)
    human_inside = _display_detected_frames(conf, model, st_frame, image)
    assert human_inside


def test_video_capture():
    yt = YouTube('https://www.youtube.com/watch?v=dQw4w9WgXcQ')
    stream = yt.streams.filter(file_extension="mp4", res=720).first()
    vid_cap = cv2.VideoCapture(stream.url)
    assert vid_cap is not None


def test_demo_video_names():
    videos = [name.split('.')[0] for name in listdir(settings.VIDEO_DIR)]
    assert videos == list(settings.VIDEOS_DICT.keys())
