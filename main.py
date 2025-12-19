import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from src import config, utils
from src.data_loader import Taxonomy, ReviewDataset
from src.silver_labeler import SilverLabeler
from src.model import DualEncoder
from src.trainer import Trainer


def main():
    utils.set_seed(config.SEED)
    print(f">>> Device: {config.DEVICE}")

    taxonomy = Taxonomy()
    tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    ckpt_path = config.MODEL_SAVE_PATH  # best_model.pt

    # ============================
    # ✅ 0) best_model.pt 있으면: 학습 스킵 → 바로 predict
    # ============================
    if os.path.exists(ckpt_path):
        print(f">>> Found checkpoint: {ckpt_path}")
        print(">>> Skip training and run predict only.")

        # 모델/트레이너 초기화 (train_loader 더미)
        model = DualEncoder(num_classes=config.NUM_CLASSES)
        dummy_loader = DataLoader([], batch_size=1, shuffle=False)
        trainer = Trainer(model, taxonomy, dummy_loader, val_loader=None, device=config.DEVICE)

        # best weights 로드
        sd = torch.load(ckpt_path, map_location=config.DEVICE)
        trainer.model.load_state_dict(sd, strict=False)
        trainer.model.eval()

        # 테스트 로더
        test_dataset = ReviewDataset(config.TEST_CORPUS_PATH, tokenizer)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS
        )

        # 예측 + 저장
        pids, predictions = trainer.predict(test_loader)
        utils.save_submission(pids, predictions, config.SUBMISSION_PATH)
        print(">>> All Done! (predict only)")
        return

    # ============================
    # 1) 없으면: Silver Label 생성 → 학습 → predict
    # ============================
    if not os.path.exists(config.SILVER_LABELS_PATH):
        print(">>> [Phase 1] Generating Silver Labels...")
        raw_dataset = ReviewDataset(config.TRAIN_CORPUS_PATH, tokenizer, silver_labels=None)
        labeler = SilverLabeler(taxonomy, raw_dataset, device=config.DEVICE)
        labeler.run()

    print(">>> Loading Training Data with Silver Labels...")
    silver_labels = torch.load(config.SILVER_LABELS_PATH)
    full_dataset = ReviewDataset(config.TRAIN_CORPUS_PATH, tokenizer, silver_labels=silver_labels)

    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"    - Train size: {len(train_dataset)}")
    print(f"    - Val size:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    print(">>> Initializing Model...")
    model = DualEncoder(num_classes=config.NUM_CLASSES)

    print(">>> [Phase 2 & 3] Start Training...")
    trainer = Trainer(model, taxonomy, train_loader, val_loader, device=config.DEVICE)
    trainer.train()

    # 학습 후 best_model.pt 로드(안전)
    if os.path.exists(ckpt_path):
        sd = torch.load(ckpt_path, map_location=config.DEVICE)
        trainer.model.load_state_dict(sd, strict=False)
        trainer.model.eval()

    print(">>> Generating Submission...")
    test_dataset = ReviewDataset(config.TEST_CORPUS_PATH, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    pids, predictions = trainer.predict(test_loader)
    utils.save_submission(pids, predictions, config.SUBMISSION_PATH)
    print(">>> All Done!")


if __name__ == "__main__":
    main()
