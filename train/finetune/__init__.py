"""Fine-tuning a pretrained image captioner on cached features.

Replaces the from-scratch CNN+LSTM pipeline, which is retained as the reported
ablation. See ``README.md`` in this package for the full procedure and the
measured numbers.

    py -3.12 -m train.finetune.cache --manifest data/manifests/vizwiz_train.jsonl
    py -3.12 -m train.finetune.train_blip --run-name blip_vizwiz --tune lora \
        --cache-index data/features/vizwiz_train_blip.jsonl \
        --cache-tensor data/features/vizwiz_train_blip.npy
    py -3.12 -m train.finetune.evaluate_blip --run-dir runs/blip_vizwiz \
        --cache-index data/features/vizwiz_val_blip.jsonl \
        --cache-tensor data/features/vizwiz_val_blip.npy
"""

__all__ = ["base", "cache", "dataset", "evaluate_blip", "train_blip"]
