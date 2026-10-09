$ErrorActionPreference = 'Stop'

$root = 'D:\project\chest-X-ray-classification'
$python = Join-Path $root '.venv\Scripts\python.exe'
$classification = Join-Path $root 'EVA-X\EVA-X\classification'
$weights = Join-Path $root 'eva_x_small_patch16_merged520k_mim.pt'
$dataRoot = 'D:\project\dataset\PNG\'
$datasetRoot = 'D:\project\dataset'
$originalTrain = Join-Path $datasetRoot 'chexpert_train_split.csv'
$originalVal = Join-Path $datasetRoot 'chexpert_val_1100.csv'
$expandedTrain = Join-Path $datasetRoot 'chexpert_train_expandedval.csv'
$expandedVal = Join-Path $datasetRoot 'chexpert_val_expanded_consolidation.csv'
$testList = Join-Path $datasetRoot 'chexpert_test_1200.csv'
$outputs = Join-Path $root 'outputs'
$queueDir = Join-Path $outputs 'autonomous_auc_7h'
$queueLog = Join-Path $queueDir 'queue.log'
$deadline = (Get-Date).AddHours(7)

New-Item -ItemType Directory -Force -Path $queueDir | Out-Null

function Log([string]$message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $message"
    Add-Content -LiteralPath $queueLog -Value $line
    Write-Output $line
}

function Invoke-Training($experiment) {
    $out = Join-Path $outputs $experiment.Name
    $console = Join-Path $out 'console.log'
    $complete = Join-Path $out 'training_complete.txt'
    New-Item -ItemType Directory -Force -Path $out | Out-Null

    if (Test-Path $complete) {
        Log "Skipping completed training: $($experiment.Name)"
        return $true
    }

    if ((Get-Date) -ge $deadline) {
        Log "Seven-hour deadline reached before $($experiment.Name)."
        return $false
    }

    Log "Starting $($experiment.Name): seed=$($experiment.Seed), pos_weights=$($experiment.Pos), train=$($experiment.TrainList)"
    Push-Location $classification
    try {
        $args = @(
            'train.py',
            '--model', 'eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE',
            '--batch_size', '16',
            '--accum_iter', '1',
            '--epochs', '5',
            '--blr', '0.0001',
            '--warmup_epochs', '1',
            '--drop_path', '0.15',
            '--input_size', '224',
            '--finetune', $weights,
            '--data_path', $dataRoot,
            '--nb_classes', '5',
            '--output_dir', $out,
            '--log_dir', $out,
            '--device', 'cuda',
            '--seed', [string]$experiment.Seed,
            '--num_workers', '8',
            '--pin_mem',
            '--train_list', $experiment.TrainList,
            '--test_list', $experiment.EvalList,
            '--eval_interval', '1',
            '--dataset', 'chexpert',
            '--light_aug',
            '--pos_weights', $experiment.Pos,
            '--chexpert_full_epoch',
            '--use_mean_pooling'
        )
        if ($experiment.ClassAug) { $args += '--consolidation_pos_aug' }
        if ($experiment.Upsample) { $args += '--chexpert_upsampling' }

        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $python @args 2>&1 | Tee-Object -FilePath $console
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previous
        if ($exitCode -ne 0) { throw "Training failed for $($experiment.Name), exit code $exitCode" }
        Set-Content -LiteralPath $complete -Value (Get-Date -Format o)
        Log "Completed training: $($experiment.Name)"
        return $true
    } finally {
        Pop-Location
    }
}

function Invoke-IndependentEval($experiment) {
    $out = Join-Path $outputs $experiment.Name
    $evalDir = Join-Path $outputs ($experiment.Name + '_test1200_eval')
    $complete = Join-Path $evalDir 'eval_complete.txt'
    if (Test-Path $complete) {
        Log "Skipping completed independent evaluation: $($experiment.Name)"
        return
    }
    $checkpoint = Join-Path $out 'checkpoint-best.pth'
    if (-not (Test-Path $checkpoint)) {
        Log "No best checkpoint for $($experiment.Name); evaluation skipped."
        return
    }
    if ((Get-Date) -ge $deadline) {
        Log "Seven-hour deadline reached before evaluation of $($experiment.Name)."
        return
    }

    New-Item -ItemType Directory -Force -Path $evalDir | Out-Null
    $console = Join-Path $evalDir 'console.log'
    Log "Evaluating $($experiment.Name) on independent 1200-image holdout."
    Push-Location $classification
    try {
        $args = @(
            'train.py', '--eval',
            '--model', 'eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE',
            '--batch_size', '16',
            '--input_size', '224',
            '--finetune', $checkpoint,
            '--data_path', $dataRoot,
            '--nb_classes', '5',
            '--output_dir', $evalDir,
            '--log_dir', $evalDir,
            '--device', 'cuda',
            '--seed', [string]$experiment.Seed,
            '--num_workers', '4',
            '--pin_mem',
            '--train_list', $experiment.TrainList,
            '--test_list', $testList,
            '--eval_interval', '1',
            '--dataset', 'chexpert',
            '--use_mean_pooling'
        )
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        & $python @args 2>&1 | Tee-Object -FilePath $console
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previous
        if ($exitCode -ne 0) { throw "Evaluation failed for $($experiment.Name), exit code $exitCode" }
        Set-Content -LiteralPath $complete -Value (Get-Date -Format o)
    } finally {
        Pop-Location
    }
}

$experiments = @(
    [pscustomobject]@{ Name = 'autonomous_orig_seed2027_w5'; TrainList = $originalTrain; EvalList = $originalVal; Seed = 2027; Pos = '1,1,5,1,1'; ClassAug = $false; Upsample = $false },
    [pscustomobject]@{ Name = 'autonomous_orig_seed2028_w3'; TrainList = $originalTrain; EvalList = $originalVal; Seed = 2028; Pos = '1,1,3,1,1'; ClassAug = $false; Upsample = $false },
    [pscustomobject]@{ Name = 'autonomous_orig_seed2029_w5_classaug'; TrainList = $originalTrain; EvalList = $originalVal; Seed = 2029; Pos = '1,1,5,1,1'; ClassAug = $true; Upsample = $false },
    [pscustomobject]@{ Name = 'autonomous_orig_seed2030_w075'; TrainList = $originalTrain; EvalList = $originalVal; Seed = 2030; Pos = '1,0.75,5,1,1'; ClassAug = $false; Upsample = $false },
    [pscustomobject]@{ Name = 'autonomous_orig_seed2031_w5_upsample'; TrainList = $originalTrain; EvalList = $originalVal; Seed = 2031; Pos = '1,1,5,1,1'; ClassAug = $true; Upsample = $true },
    [pscustomobject]@{ Name = 'autonomous_expanded_seed2032_w3'; TrainList = $expandedTrain; EvalList = $expandedVal; Seed = 2032; Pos = '1,1,3,1,1'; ClassAug = $false; Upsample = $false },
    [pscustomobject]@{ Name = 'autonomous_expanded_seed2033_w5_classaug'; TrainList = $expandedTrain; EvalList = $expandedVal; Seed = 2033; Pos = '1,1,5,1,1'; ClassAug = $true; Upsample = $false }
)

Log "Queue started. Deadline: $($deadline.ToString('s'))"
foreach ($experiment in $experiments) {
    if (-not (Invoke-Training $experiment)) { break }
    Invoke-IndependentEval $experiment
}
Log 'Queue finished or deadline reached.'
