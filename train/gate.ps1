# Quality gate pipeline. Run after the captioner pipeline finishes so the two
# do not contend for the GPU.

$py = 'C:\Users\Lain\AppData\Local\Programs\Python\Python312\python.exe'
$log = "$env:TEMP\gate.log"
$err = "$env:TEMP\gate_err.log"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $msg
    $line | Tee-Object -FilePath $log -Append
}

Log "=== QUALITY GATE: OpenCV baseline (comparison the proposal promises) ==="
& $py -m train.quality.train_gate --baseline *>> $log 2>> $err
Log "BASELINE_EXIT=$LASTEXITCODE"

Log "=== QUALITY GATE: trained head on cached features ==="
& $py -m train.quality.train_gate --epochs 25 *>> $log 2>> $err
Log "GATE_EXIT=$LASTEXITCODE"

Log "=== QUALITY GATE PIPELINE COMPLETE ==="
