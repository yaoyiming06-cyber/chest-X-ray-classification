$ErrorActionPreference = 'Stop'

$root = 'D:\project\chest-X-ray-classification'
$python = Join-Path $root '.venv\Scripts\python.exe'
$classification = Join-Path $root 'EVA-X\EVA-X\classification'
$dataRoot = 'D:\project\dataset\PNG\'
$trainList = 'D:\project\dataset\chexpert_train_expandedval.csv'
$valList = 'D:\project\dataset\chexpert_val_expanded_consolidation.csv'
$weights = Join-Path $root 'eva_x_small_patch16_merged520k_mim.pt'
$outputs = Join-Path $root 'outputs'
$queueDir = Join-Path $outputs 'resize_pad_queue'
$queueLog = Join-Path $queueDir 'queue.log'
$deadline = (Get-Date).AddHours(4)

New-Item -ItemType Directory -Force -Path $queueDir | Out-Null

function Log([string]$message) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $message"
    Add-Content -LiteralPath $queueLog -Value $line
    Write-Output $line
}

function Invoke-Training($experiment) {
    if ((Get-Date) -ge $deadline) {
        Log "Deadline reached before $($experiment.Name)."
        return $false
    }

    $out = Join-Path $outputs $experiment.Name
    $console = Join-Path $out 'console.log'
    $complete = Join-Path $out 'training_complete.txt'
    New-Item -ItemType Directory -Force -Path $out | Out-Null
    if (Test-Path $complete) {
        Log "Skipping completed training: $($experiment.Name)"
        return $true
    }

    Log "Starting $($experiment.Name): seed=$($experiment.Seed), pos_weights=$($experiment.Pos)"
    Push-Location $classification
    try {
        $args = @(
            'train.py',
            '--model', 'eva02_small_patch16_xattn_fusedLN_SwiGLU_preln_RoPE',
            '--batch_size', '16',
            '--accum_iter', '1',
            '--input_size', '224',
            '--epochs', '5',
            '--blr', '0.0001',
            '--warmup_epochs', '1',
            '--drop_path', '0.15',
            '--layer_decay', '0.75',
            '--finetune', $weights,
            '--data_path', $dataRoot,
            '--nb_classes', '5',
            '--output_dir', $out,
            '--log_dir', $out,
            '--device', 'cuda',
            '--seed', [string]$experiment.Seed,
            '--num_workers', '8',
            '--pin_mem',
            '--train_list', $trainList,
            '--test_list', $valList,
            '--eval_interval', '1',
            '--dataset', 'chexpert',
            '--use_mean_pooling',
            '--pos_weights', $experiment.Pos,
            '--consolidation_pos_aug',
            '--chexpert_full_epoch',
            '--aug_strategy', 'light',
            '--light_aug',
            '--resize_mode', 'pad'
        )
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & $python @args 2>&1 | Tee-Object -FilePath $console
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        if ($exitCode -ne 0) {
            throw "Training failed for $($experiment.Name), exit code $exitCode"
        }
        Set-Content -LiteralPath $complete -Value (Get-Date -Format o)
        Log "Completed training: $($experiment.Name)"
        return $true
    }
    finally {
        Pop-Location
    }
}

$experiments = @(
    [pscustomobject]@{ Name = 'pad_light_seed2033_w5'; Seed = 2033; Pos = '1,1,5,1,1' },
    [pscustomobject]@{ Name = 'pad_light_seed2034_w5'; Seed = 2034; Pos = '1,1,5,1,1' },
    [pscustomobject]@{ Name = 'pad_light_seed2033_w3'; Seed = 2033; Pos = '1,1,3,1,1' }
)

Log "Queue started. Deadline: $($deadline.ToString('s'))"
foreach ($experiment in $experiments) {
    if (-not (Invoke-Training $experiment)) { break }
}
Log 'Queue finished or deadline reached.'
