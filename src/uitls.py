import os
import random
import numpy as np
import torch
import csv
import logging
from . import config  # 같은 폴더 내 config.py 임포트

def set_seed(seed=42):
    """
    [필수] 재현성을 위해 모든 라이브러리의 랜덤 시드를 고정합니다.
    프로젝트 시작 시 가장 먼저 호출해야 합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # CuDNN의 랜덤성 제어 (속도는 느려질 수 있지만 재현성은 보장)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Info] Random Seed set to {seed}")

def get_logger(name, log_dir=None):
    """
    학습 과정을 콘솔과 파일에 동시에 기록하는 로거를 생성합니다.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log'))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def save_submission(pids, predictions, filename):
    """
    [필수] Kaggle 제출용 CSV 파일을 생성합니다.
    pids: 제품 ID 리스트
    predictions: 예측된 라벨 리스트 (각 요소는 [1, 23, 50] 형태의 리스트)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pid", "labels"])  # 헤더
        
        for pid, labels in zip(pids, predictions):
            # labels가 텐서라면 리스트로 변환
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().tolist()
            
            # 공백으로 구분하여 문자열로 변환 (예: "1 23 50")
            label_str = " ".join(map(str, sorted(labels)))
            writer.writerow([pid, label_str])
            
    print(f"[Info] Submission file saved to {filename}")

def clean_text(text):
    """
    리뷰 텍스트 전처리 (필요 시 수정 가능)
    """
    return text.strip().replace("\n", " ")