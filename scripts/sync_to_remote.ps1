param(
    [Parameter(Mandatory = $true)]
    [string]$Target,

    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path
)

$items = @(
    "configs",
    "data\samples",
    "docs",
    "scripts",
    "src",
    "README.md",
    "requirements.txt"
)

foreach ($item in $items) {
    $path = Join-Path $ProjectRoot $item
    if (Test-Path $path) {
        scp -r $path $Target
    }
}
