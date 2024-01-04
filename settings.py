import sys
from pathlib import Path

file_path = Path(__file__).resolve()
root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))


ROOT = root_path.relative_to(Path.cwd())

DEMOVIDEO = 'DemoVideo'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'
MYVIDEO = 'MyVideo'

SOURCES_LIST = [DEMOVIDEO, RTSP, YOUTUBE, MYVIDEO]

VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'cat_and_human': VIDEO_DIR / 'cat_and_human.mp4',
    'only_cat': VIDEO_DIR / 'only_cat.mp4',
    'thief': VIDEO_DIR / 'thief.mp4',
}

TEST_PIC = ROOT / 'picture_for_test' / 'test_human.jpg'

MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8n.pt'
