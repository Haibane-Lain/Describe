# Full end-to-end attention captioner run, detached from the DSH session.
#
# Stage 1: train encoder + decoder jointly on Flickr30k, selecting on val BLEU-4.
# Stage 2: evaluate the best checkpoint on the held-out split.
#
# Detached so it survives session interruption. Every epoch writes last.pt and
# every 5th writes epoch_NNN.pt, so nothing is lost if it is killed.

$py = 'C:\Users\Lain\AppData\Local\Programs\Python\Python312\python.exe'
$root = 'C:\Coding\General Image Recognizer'
$logDir = Join-Path $root 'train\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$trainOut = Join-Path $logDir 'e2e_train.log'
$trainErr = Join-Path $logDir 'e2e_train_err.log'
$evalOut = Join-Path $logDir 'e2e_eval.log'
$evalErr = Join-Path $logDir 'e2e_eval_err.log'

# ---------------------------------------------------------------- stage 1
$trainArgs = @(
    '-u', '-m', 'train.train.train_e2e',
    '--manifest', 'data/manifests/flickr30k.jsonl',
    '--image-root', 'data/raw/flickr30k/images',
    '--run-name', 'flickr30k_e2e',
    '--backbone', 'resnet50', '--image-size', '224',
    '--epochs', '40', '--batch-size', '64',
    '--lr', '5e-4', '--cnn-lr', '1e-5', '--dropout', '0.5',
    '--max-caption-len', '32', '--scheduled-sampling', '0.3',
    '--holdout', '0.05', '--val-bleu-every', '2', '--val-bleu-limit', '200',
    '--beam-size', '5', '--min-len', '2',
    '--repetition-penalty', '1.2', '--no-repeat-ngram', '3',
    '--length-alpha', '0.7', '--amp', '--log-every', '200'
)

$train = Start-Process -FilePath $py -ArgumentList $trainArgs -WorkingDirectory $root `
    -RedirectStandardOutput $trainOut -RedirectStandardError $trainErr `
    -WindowStyle Hidden -PassThru
$train.WaitForExit()
"TRAIN_EXIT=$($train.ExitCode)" | Out-File -FilePath $trainOut -Append -Encoding utf8

# ---------------------------------------------------------------- stage 2
$evalArgs = @(
    '-u', '-m', 'train.train.evaluate',
    '--checkpoint', 'checkpoints/flickr30k_e2e/best.pt',
    '--feature-index', 'data/features/flickr30k_resnet101_spatial.jsonl',
    '--manifest', 'data/manifests/flickr30k_holdout.jsonl',
    '--beam-size', '5', '--min-len', '2',
    '--repetition-penalty', '1.2', '--no-repeat-ngram', '3',
    '--out', 'runs/flickr30k_e2e/eval_holdout.json',
    '--pred-out', 'runs/flickr30k_e2e/preds_holdout.jsonl'
)

$eval = Start-Process -FilePath $py -ArgumentList $evalArgs -WorkingDirectory $root `
    -RedirectStandardOutput $evalOut -RedirectStandardError $evalErr `
    -WindowStyle Hidden -PassThru
$eval.WaitForExit()
"EVAL_EXIT=$($eval.ExitCode)" | Out-File -FilePath $evalOut -Append -Encoding utf8
"E2E PIPELINE COMPLETE" | Out-File -FilePath $evalOut -Append -Encoding utf8
