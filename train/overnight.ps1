# Overnight training pipeline: Flickr30k pretrain -> VizWiz fine-tune -> ablation.
# Each stage is independent; a failure in one does not stop the others.

$py = 'C:\Users\Lain\AppData\Local\Programs\Python\Python312\python.exe'
$log = "$env:TEMP\overnight.log"
$err = "$env:TEMP\overnight_err.log"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    $line | Tee-Object -FilePath $log -Append
}

Log "=== STAGE 1: Flickr30k pretrain, 60 epochs, scheduled sampling ==="
& $py -m train.train.train_captioner `
    --manifest data/manifests/flickr30k.jsonl `
    --run-name flickr30k_ss --epochs 60 --holdout 0.05 `
    --scheduled-sampling 0.5 --log-every 400 `
    *>> $log 2>> $err
Log "STAGE1_EXIT=$LASTEXITCODE"

Log "=== STAGE 2: VizWiz fine-tune from stage 1 ==="
& $py -m train.train.train_captioner `
    --manifest data/manifests/vizwiz_train.jsonl `
    --val-manifest data/manifests/vizwiz_val.jsonl `
    --feature-index data/features/vizwiz_train_resnet101.jsonl `
    --run-name vizwiz_ft --epochs 25 --lr 1e-4 `
    --scheduled-sampling 0.4 --log-every 200 `
    --init-from checkpoints/flickr30k_ss/best.pt `
    *>> $log 2>> $err
Log "STAGE2_EXIT=$LASTEXITCODE"

Log "=== STAGE 3: VizWiz from scratch (domain-shift ablation) ==="
& $py -m train.train.train_captioner `
    --manifest data/manifests/vizwiz_train.jsonl `
    --val-manifest data/manifests/vizwiz_val.jsonl `
    --feature-index data/features/vizwiz_train_resnet101.jsonl `
    --run-name vizwiz_scratch --epochs 25 --lr 1e-3 `
    --scheduled-sampling 0.5 --log-every 200 `
    *>> $log 2>> $err
Log "STAGE3_EXIT=$LASTEXITCODE"

Log "=== EVALUATION: all three runs on VizWiz validation ==="
foreach ($run in @('flickr30k_ss', 'vizwiz_ft', 'vizwiz_scratch')) {
    if (Test-Path "checkpoints/$run/best.pt") {
        Log "evaluating $run"
        & $py -m train.train.evaluate `
            --checkpoint "checkpoints/$run/best.pt" `
            --feature-index data/features/vizwiz_val_resnet101.jsonl `
            --manifest data/manifests/vizwiz_val.jsonl `
            --beam-size 5 --min-len 2 `
            --repetition-penalty 1.3 --no-repeat-ngram 3 `
            --out "runs/$run/eval_vizwiz_val.json" `
            --pred-out "runs/$run/preds_vizwiz_val.jsonl" `
            *>> $log 2>> $err
        Log "EVAL_${run}_EXIT=$LASTEXITCODE"
    }
    else {
        Log "SKIP $run (no checkpoint)"
    }
}

Log "=== OVERNIGHT PIPELINE COMPLETE ==="
