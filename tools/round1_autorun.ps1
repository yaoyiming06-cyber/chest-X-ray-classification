$ErrorActionPreference = 'Stop'

$root = 'D:\project\chest-X-ray-classification'
$python = Join-Path $root '.venv\Scripts\python.exe'
$classification = Join-Path $root 'EVA-X\EVA-X\classification'
$weights = Join-Path $root 'eva_x_small_patch16_merged520k_mim.pt'
$dataRoot = 'D:\project\dataset\PNG\'
$trainList = 'D:\project\dataset\chexpert_train_split.csv'
$valList = 'D:\project\dataset\chexpert_val_1100.csv'
$testList = 'D:\project\dataset\chexpert_test_1200.csv'
$outputs = Join-Path $root 'outputs'
$queueLog = Join-Path $outputs 'round1_autorun.log'
$baseline = Join-Path $outputs 'round1_full_w5'

function Log([string]$message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $message"
    Add-Content -LiteralPath $queueLog -Value $line
}

New-Item -ItemType Directory -Force -Path $outputs | Out-Null
Log 'Queue started.'
Log 'Waiting for round1_full_w5 to write max_auc.'
while (-not (Test-Path (Join-Path $baseline 'log.txt')) -or
       -not (Select-String -LiteralPath (Join-Path $baseline 'log.txt') -Pattern '^max_auc=' -Quiet)) {
    Start-Sleep -Seconds 30
}
Start-Sleep -Seconds 30
Log 'round1_full_w5 is complete.'

function Run-Eval([string]$name, [string]$checkpointDir) {
    $evalDir = Join-Path $outputs ($name + '_test1200_eval')
    if (Test-Path (Join-Path $evalDir 'eval_complete.txt')) {
        Log "$name independent evaluation already complete; skipping."
        return
    }
    New-Item -ItemType Directory -Force -Path $evalDir | Out-Null
    $checkpoint = Join-Path $checkpointDir 'checkpoint-best.pth'
    $console = Join-Path $evalDir 'console.log'
    Log "Evaluating $name on test1200."
    Push-Location $classification
    try {
        $args = @(
            'train.py', '--eval', '--model', 'eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE',
            '--batch_size', '8', '--input_size', '224', '--finetune', $checkpoint,
            '--data_path', $dataRoot, '--nb_classes', '5', '--output_dir', $evalDir,
            '--log_dir', $evalDir, '--device', 'cuda', '--seed', '2026',
            '--num_workers', '2', '--pin_mem', '--train_list', $trainList,
            '--test_list', $testList, '--eval_interval', '1', '--dataset', 'chexpert',
            '--use_mean_pooling'
        )
        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $python @args 2>&1 | Tee-Object -FilePath $console
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorAction
        if ($exitCode -ne 0) { throw "Evaluation failed for $name with exit code $exitCode" }
        Set-Content -LiteralPath (Join-Path $evalDir 'eval_complete.txt') -Value (Get-Date -Format o)
    } finally {
        Pop-Location
    }
}

Run-Eval 'round1_full_w5' $baseline

$experiments = @(
    @{ Name = 'round1_full_w5_classaug_pool'; Pos = '1,1,5,1,1'; ClassAug = $true },
    @{ Name = 'round1_full_w8_pool'; Pos = '1,1,8,1,1'; ClassAug = $false },
    @{ Name = 'round1_full_w8_classaug_pool'; Pos = '1,1,8,1,1'; ClassAug = $true },
    @{ Name = 'round1_full_w8_pleural115_pool'; Pos = '1,1,8,1,1.15'; ClassAug = $true }
)

foreach ($experiment in $experiments) {
    $name = $experiment.Name
    $out = Join-Path $outputs $name
    $console = Join-Path $out 'console.log'
    if (-not (Test-Path (Join-Path $out 'log.txt'))) {
        New-Item -ItemType Directory -Force -Path $out | Out-Null
        Log "Training $name with pos_weights=$($experiment.Pos), classaug=$($experiment.ClassAug)."
        Push-Location $classification
        try {
            $args = @(
                'train.py', '--model', 'eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE',
                '--batch_size', '8', '--accum_iter', '2', '--epochs', '3', '--blr', '0.0001',
                '--warmup_epochs', '1', '--drop_path', '0.15', '--input_size', '224',
                '--finetune', $weights, '--data_path', $dataRoot, '--nb_classes', '5',
                '--output_dir', $out, '--log_dir', $out, '--device', 'cuda', '--seed', '2026',
                '--num_workers', '4', '--pin_mem', '--train_list', $trainList,
                '--test_list', $valList, '--eval_interval', '1', '--dataset', 'chexpert',
                '--light_aug', '--pos_weights', $experiment.Pos, '--chexpert_full_epoch',
                '--use_mean_pooling'
            )
            if ($experiment.ClassAug) { $args += '--consolidation_pos_aug' }
            $previousErrorAction = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            & $python @args 2>&1 | Tee-Object -FilePath $console
            $exitCode = $LASTEXITCODE
            $ErrorActionPreference = $previousErrorAction
            if ($exitCode -ne 0) { throw "Training failed for $name with exit code $exitCode" }
        } finally {
            Pop-Location
        }
    } else {
        Log "$name training log already exists; skipping training."
    }
    if (Test-Path (Join-Path $out 'checkpoint-best.pth')) {
        Run-Eval $name $out
    } else {
        Log "No checkpoint-best.pth for $name; independent evaluation skipped."
    }
}

Log 'Queue finished.'
