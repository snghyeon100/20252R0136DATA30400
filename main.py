import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from src import config, utils
from src.data_loader import Taxonomy, ReviewDataset
from src.silver_labeler import SilverLabeler
from src.model import DualEncoder
from src.early_stop import Trainer

def main():
    # 1. 초기 설정
    utils.set_seed(config.SEED)
    print(f">>> Device: {config.DEVICE}")
    
    # 2. 데이터 준비 (Taxonomy & Dataset)
    taxonomy = Taxonomy()
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    
    # Silver Label이 없으면 먼저 생성 (Phase 1)
    if not os.path.exists(config.SILVER_LABELS_PATH):
        print(">>> [Phase 1] Generating Silver Labels...")
        # SilverLabeler용 데이터셋 (라벨 없이 텍스트만 로드)
        raw_dataset = ReviewDataset(config.TRAIN_CORPUS_PATH, tokenizer, silver_labels=None)
        labeler = SilverLabeler(taxonomy, raw_dataset, device=config.DEVICE)
        labeler.run()
    
    # 3. 학습용 데이터셋 로드 (Silver Label 포함)
    print(">>> Loading Training Data with Silver Labels...")
    # 저장된 Silver Label 파일을 로드해서 Dataset에 주입
    silver_labels = torch.load(config.SILVER_LABELS_PATH)
    full_dataset = ReviewDataset(config.TRAIN_CORPUS_PATH, tokenizer, silver_labels=silver_labels)
    
    # [핵심] Train / Validation 분할 (9:1)
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"    - Train size: {len(train_dataset)}")
    print(f"    - Val size:   {len(val_dataset)}")
    
    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    # 4. 모델 초기화
    print(">>> Initializing Model...")
    model = DualEncoder(num_classes=config.NUM_CLASSES)
    
    # 5. 학습 (Phase 2 & 3)
    print(">>> [Phase 2 & 3] Start Training...")
    trainer = Trainer(model, taxonomy, train_loader, val_loader, device=config.DEVICE)
    trainer.train() # Early Stopping이 포함된 학습 시작
    
    # 6. 최종 예측 (Kaggle 제출용)
    print(">>> Generating Submission...")
    # Best Model 로드
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    
    # 테스트 데이터셋 로드
    test_dataset = ReviewDataset(config.TEST_CORPUS_PATH, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    # 예측 수행
    pids, predictions = trainer.predict(test_loader)
    
    # CSV 저장
    utils.save_submission(pids, predictions, config.SUBMISSION_PATH)
    print(">>> All Done!")

if __name__ == "__main__":
    main()