import os

# module/path_header.py 절대 경로
FILE = os.path.abspath(__file__)

# root 디렉토리: Mullin/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(FILE)))

MODEL_DIR = os.path.join(ROOT_DIR, 'model')
RAW_DIR = os.path.join(ROOT_DIR, 'raw')
PREPROCESS_DIR = os.path.join(ROOT_DIR, 'preprocess')
PREDICT_DIR = os.path.join(ROOT_DIR, 'predict')
