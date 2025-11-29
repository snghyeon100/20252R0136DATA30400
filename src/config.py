import os
import torch

# ==========================================
# 1. Reproducibility (재현성 필수 설정)
# ==========================================
SEED = 42

# ==========================================
# 2. File Paths (경로 설정)
# ==========================================
# 현재 파일(src/config.py)의 상위 폴더를 프로젝트 루트로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 및 출력 폴더

OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 원본 데이터 경로 (압축 푼 폴더명에 따라 수정 필요할 수 있음)
# 예: data/Amazon_products/train/train_corpus.txt
RAW_DATA_DIR = os.path.join(BASE_DIR, "Amazon_products")
TRAIN_CORPUS_PATH = os.path.join(RAW_DATA_DIR, "train", "train_corpus.txt")
TEST_CORPUS_PATH = os.path.join(RAW_DATA_DIR, "test", "test_corpus.txt")

# 메타 데이터 경로
TAXONOMY_PATH = os.path.join(RAW_DATA_DIR, "class_hierarchy.txt")
KEYWORDS_PATH = os.path.join(RAW_DATA_DIR, "class_related_keywords.txt")

# 생성 파일 저장 경로 (중간 결과물)
EXPANDED_KEYWORDS_PATH = os.path.join(OUTPUT_DIR, "keywords_expanded.json")
SIMILARITY_MATRIX_PATH = os.path.join(OUTPUT_DIR, "similarity_scores.pt")
SILVER_LABELS_PATH = os.path.join(OUTPUT_DIR, "silver_labels.pt")

# 모델 저장 및 제출 파일
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(BASE_DIR, "submission.csv") # 학번_final.csv로 나중에 변경

# ==========================================
# 3. Model & Data Settings
# ==========================================
NUM_CLASSES = 531
MAX_SEQ_LEN = 128        # BERT 입력 최대 길이 (메모리 부족 시 줄임)
BERT_MODEL_NAME = "bert-base-uncased"

# Phase 1: Silver Label 생성용 NLI 모델 (논문 설정)
NLI_MODEL_NAME = "roberta-large-mnli" 

# ==========================================
# 4. Training Hyperparameters (논문 Section 4 참조)
# ==========================================
BATCH_SIZE = 64
NUM_EPOCHS = 5           # 전체 에폭 (Phase 1 + Phase 2 적절히 배분)

# --- Phase 1: Initial Training Learning Rates ---
LR_BERT_P1 = 5e-5        # BERT 파트 (조금만 수정)
LR_BASE_P1 = 4e-3        # GNN 및 분류기 파트 (크게 수정)

# --- Phase 2: Self-Training Learning Rates ---
LR_BERT_P2 = 1e-6        # BERT 파트 (아주 미세하게)
LR_BASE_P2 = 5e-4        # 나머지 파트
ST_UPDATE_INTERVAL = 25  # Self-Training 타겟(Q) 업데이트 주기 (배치 단위)

# ==========================================
# 5. Hardware Settings
# ==========================================
# GPU가 있으면 사용, 없으면 CPU (Mac은 'mps' 가능하지만 안전하게 cpu/cuda 추천)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2          # 데이터 로더 병렬 처리 개수 (Windows라면 0 추천)

# ==========================================
# 6. Prompt Ensembles (Silver Label 생성용)
# ==========================================
# 다양한 관점에서 질문하여 노이즈를 줄임
PROMPT_TEMPLATES = [
    "This product is about {}.",         # 논문 기본 (Baseline)
    "Category: {}.",                    # 짧은 형식
    "It belongs to the {} category.",   # 계층 구조 강조
    "A review of {}.",                  # 도메인(리뷰) 강조
    "The topic of this text is {}."     # 일반 주제
]

# 디렉토리 자동 생성 (없으면 에러 나니까)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RAW_DATA_DIR, exist_ok=True)