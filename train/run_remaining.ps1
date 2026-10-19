# Remaining pipeline, detached from the DSH session.
#
# Waits for the current attention training to finish, then:
#   1. evaluates the Flickr30k attention model on held-out data
#   2. fine-tunes it on VizWiz-Captions (the domain that matters)
#   3. evaluates the fine-tuned model on VizWiz validation
#   4. evaluates the VizWiz-from-scratch ablation
#
# Every stage writes its own metrics file, so partial completion is still
# useful. Safe to re-run: stages skip when their output already exists.

$py = 'C:\Users\Lain\AppData\Local\Programs\Python\Python312\python.exe'
$root = 'C:\Coding\General Image Recognizer'
$logDir = Join-Path $root 'train\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'pipeline.log'

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $msg
    $line | Out-File -FilePath $log -Append -Encoding utf8
    Write-Output $line
}

Log "pipeline started; waiting for flickr30k_attn training to finish"

# ---------------------------------------------------------------- 1. wait
$deadline = (Get-Date).AddHours(6)
while ((Get-Date) -lt $deadline) {
    $running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -match 'train_captioner' -and $_.CommandLine -match 'flickr30k_attn' }
    if (-not $running) { break }
    Start-Sleep -Seconds 60
}
Log "attention training no longer running"

# ------------------------------------------------- 2. held-out evaluation
$evalOut = Join-Path $root 'runs\flickr30k_attn\eval_holdout_final.json'
Set-Location $root
& $py -m train.train.evaluate `
    --checkpoint checkpoints/flickr30k_attn/best.pt `
    --feature-index data/features/flickr30k_resnet101_spatial.jsonl `
    --manifest data/manifests/flickr30k_holdout.jsonl `
    --beam-size 5 --min-len 2 `
    --repetition-penalty 1.2 --no-repeat-ngram 3 `
    --out $evalOut *>> $log 2>&1
Log "FLICKR_EVAL_EXIT=$LASTEXITCODE"

# ------------------------------------------- 3. VizWiz fine-tune (stage 2)
$ftDir = Join-Path $root 'checkpoints\vizwiz_attn_ft'
if (-not (Test-Path (Join-Path $ftDir 'vocab.pkl'))) {
    Log "fine-tuning on VizWiz-Captions"
    & $py -u -m train.train.train_captioner `
        --manifest data/manifests/vizwiz_train.jsonl `
        --val-manifest data/manifests/vizwiz_val.jsonl `
        --model attention `
        --run-name vizwiz_attn_ft --epochs 12 --lr 1e-4 --dropout 0.4 `
        --batch-size 128 --scheduled-sampling 0.3 --log-every 200 `
        --init-from checkpoints/flickr30k_attn/best.pt *>> $log 2>&1
    Log "FT_EXIT=$LASTEXITCODE"
} else {
    Log "fine-tune checkpoint already exists; skipping"
}

# ------------------------------------- 4. evaluate fine-tuned on VizWiz
if (Test-Path (Join-Path $ftDir 'best.pt')) {
    & $py -m train.train.evaluate `
        --checkpoint checkpoints/vizwiz_attn_ft/best.pt `
        --feature-index data/features/vizwiz_val_resnet101_spatial.jsonl `
        --manifest data/manifests/vizwiz_val.jsonl `
        --beam-size 5 --min-len 2 `
        --repetition-penalty 1.2 --no-repeat-ngram 3 `
        --out runs/vizwiz_attn_ft/eval_vizwiz_val.json `
        --pred-out runs/vizwiz_attn_ft/preds_vizwiz_val.jsonl *>> $log 2>&1
    Log "FT_EVAL_EXIT=$LASTEXITCODE"
}

# ------------------------------------- 5. VizWiz from scratch (ablation)
$scratchDir = Join-Path $root 'checkpoints\vizwiz_attn_scratch'
if (-not (Test-Path (Join-Path $scratchDir 'vocab.pkl'))) {
    Log "training VizWiz ablation from scratch"
    & $py -u -m train.train.train_captioner `
        --manifest data/manifests/vizwiz_train.jsonl `
        --val-manifest data/manifests/vizwiz_val.jsonl `
        --model attention `
        --run-name vizwiz_attn_scratch --epochs 12 --lr 5e-4 --dropout 0.4 `
        --batch-size 128 --scheduled-sampling 0.4 --log-every 200 *>> $log 2>&1
    Log "SCRATCH_EXIT=$LASTEXITCODE"
}

if (Test-Path (Join-Path $scratchDir 'best.pt')) {
    & $py -m train.train.evaluate `
        --checkpoint checkpoints/vizwiz_attn_scratch/best.pt `
        --feature-index data/features/vizwiz_val_resnet101_spatial.jsonl `
        --manifest data/manifests/vizwiz_val.jsonl `
        --beam-size 5 --min-len 2 `
        --repetition-penalty 1.2 --no-repeat-ngram 3 `
        --out runs/vizwiz_attn_scratch/eval_vizwiz_val.json *>> $log 2>&1
    Log "SCRATCH_EVAL_EXIT=$LASTEXITCODE"
}

Log "PIPELINE COMPLETE"
