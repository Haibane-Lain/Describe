# Continue attention training to completion, detached from the DSH session.
# Uses Start-Process so the training survives session/job interruption.

$py = 'C:\Users\Lain\AppData\Local\Programs\Python\Python312\python.exe'
$root = 'C:\Coding\General Image Recognizer'
$log = Join-Path $root 'train\logs\attn_continue.log'
$err = Join-Path $root 'train\logs\attn_continue_err.log'

New-Item -ItemType Directory -Force -Path (Join-Path $root 'train\logs') | Out-Null

$args = @(
    '-u', '-m', 'train.train.train_captioner',
    '--manifest', 'data/manifests/flickr30k.jsonl',
    '--model', 'attention',
    '--run-name', 'flickr30k_attn',
    '--epochs', '14',
    '--holdout', '0.05',
    '--batch-size', '128',
    '--lr', '5e-4',
    '--dropout', '0.4',
    '--scheduled-sampling', '0.4',
    '--log-every', '300',
    '--resume', 'checkpoints/flickr30k_attn/last.pt'
)

Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $root `
    -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden

Write-Output "launched detached attention training continuation"
Write-Output "stdout: $log"
Write-Output "stderr: $err"
