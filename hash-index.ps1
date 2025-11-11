param(
  [string]$Path = ".",
  [ValidateSet('SHA256','SHA1','SHA384','SHA512')]
  [string]$Algorithm = 'SHA256',
  [string]$IndexFile = 'SHA256SUMS.txt',
  [switch]$SkipExisting,          # Skip files already listed in the index
  [switch]$UseSidecar,            # Reuse sidecar hashes like file.ext.sha256 / .sha1
  [switch]$Verify,                # Verify files against the index instead of writing
  [string[]]$Exclude = @('.git','node_modules',"$IndexFile","$IndexFile.sha256")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info([string]$msg) { Write-Host "[hash-index] $msg" }

function Get-ExpectedDigestLength([string]$alg) {
  switch ($alg.ToUpper()) {
    'SHA1'   { 40 }
    'SHA256' { 64 }
    'SHA384' { 96 }
    'SHA512' { 128 }
    default  { throw "Unsupported algorithm: $alg" }
  }
}

function Get-SidecarHash([string]$filePath, [string]$alg) {
  $algLower = $alg.ToLower()
  $candidates = @("$filePath.$algLower")
  # Common alternates
  if ($alg -eq 'SHA1') { $candidates += "$filePath.sha1" }
  if ($alg -eq 'SHA256') { $candidates += "$filePath.sha256" }
  if ($alg -eq 'SHA384') { $candidates += "$filePath.sha384" }
  if ($alg -eq 'SHA512') { $candidates += "$filePath.sha512" }
  foreach ($c in $candidates) {
    if (Test-Path -LiteralPath $c) {
      $content = Get-Content -LiteralPath $c -Raw
      # Try to parse common formats: "<hash>  filename" or just "<hash>"
      $m = [regex]::Match($content, '([A-Fa-f0-9]{40,128})')
      if ($m.Success) { return $m.Groups[1].Value.ToLower() }
    }
  }
  return $null
}

function Parse-Index([string]$indexPath, [string]$alg) {
  $map = @{}
  if (-not (Test-Path -LiteralPath $indexPath)) { return $map }
  $len = Get-ExpectedDigestLength $alg
  $lines = Get-Content -LiteralPath $indexPath
  foreach ($line in $lines) {
    if (-not $line) { continue }
    $m = [regex]::Match($line, "^([A-Fa-f0-9]{$len})\s\s(.*)$")
    if ($m.Success) {
      $h = $m.Groups[1].Value.ToLower()
      $p = $m.Groups[2].Value
      # Normalize stored path
      $map[$p] = $h
    }
  }
  return $map
}

function Format-IndexLine([string]$hash, [string]$relPath) {
  # Use sha256sum-like format: "<hash><two spaces><path>"
  return "$hash  $relPath"
}

function Get-RelativePath([string]$root, [string]$full) {
  $uriRoot = (Resolve-Path -LiteralPath $root).Path
  $uriFile = (Resolve-Path -LiteralPath $full).Path
  $rel = [System.IO.Path]::GetRelativePath($uriRoot, $uriFile)
  return $rel
}

function Should-Exclude([string]$root, [string]$full, [string[]]$patterns) {
  $rel = Get-RelativePath $root $full
  foreach ($pat in $patterns) {
    if ([string]::IsNullOrWhiteSpace($pat)) { continue }
    # Treat pattern as folder/file prefix match or file name match
    if ($rel -like "$pat*" -or ([System.IO.Path]::GetFileName($rel) -like $pat)) { return $true }
  }
  return $false
}

$root = Resolve-Path -LiteralPath $Path
$indexPath = Join-Path -Path $root -ChildPath $IndexFile

# If algorithm doesn't match index filename, align the default name
if ($PSBoundParameters.ContainsKey('Algorithm') -and -not $PSBoundParameters.ContainsKey('IndexFile')) {
  switch ($Algorithm) {
    'SHA1'   { $indexPath = Join-Path $root 'SHA1SUMS.txt' }
    'SHA384' { $indexPath = Join-Path $root 'SHA384SUMS.txt' }
    'SHA512' { $indexPath = Join-Path $root 'SHA512SUMS.txt' }
    default  { }
  }
}

# Load existing index entries
$existing = Parse-Index $indexPath $Algorithm
if ($Verify) {
  if (-not (Test-Path -LiteralPath $indexPath)) { throw "Index file not found: $indexPath" }
  Write-Info "Verifying files listed in $(Split-Path -Leaf $indexPath)"
  $fail = 0
  foreach ($kv in $existing.GetEnumerator()) {
    $rel = $kv.Key
    $expected = $kv.Value
    $full = Join-Path $root $rel
    if (-not (Test-Path -LiteralPath $full)) {
      Write-Host "MISSING  $rel" -ForegroundColor Yellow
      $fail++
      continue
    }
    $h = (Get-FileHash -Algorithm $Algorithm -LiteralPath $full).Hash.ToLower()
    if ($h -eq $expected) {
      Write-Host "OK       $rel" -ForegroundColor Green
    } else {
      Write-Host "MISMATCH $rel" -ForegroundColor Red
      $fail++
    }
  }
  if ($fail -gt 0) { throw "$fail file(s) failed verification" } else { Write-Info "Verification succeeded" }
  return
}

Write-Info "Scanning '$root' for files to hash ($Algorithm)"

# Build set of files to hash
$all = Get-ChildItem -LiteralPath $root -Recurse -File -Force
$toHash = @()
foreach ($f in $all) {
  if (Should-Exclude $root $f.FullName $Exclude) { continue }
  # Never hash the index or its sidecar
  if ($f.FullName -ieq $indexPath) { continue }
  if ($f.FullName -ieq "$indexPath.$($Algorithm.ToLower())") { continue }
  $rel = Get-RelativePath $root $f.FullName
  if ($SkipExisting -and $existing.ContainsKey($rel)) { continue }
  $toHash += $f
}

Write-Info ("Found {0} file(s) to hash" -f $toHash.Count)

# Compute hashes
$newEntries = @{}
foreach ($f in $toHash) {
  $rel = Get-RelativePath $root $f.FullName
  $hash = $null
  if ($UseSidecar) { $hash = Get-SidecarHash $f.FullName $Algorithm }
  if (-not $hash) { $hash = (Get-FileHash -Algorithm $Algorithm -LiteralPath $f.FullName).Hash.ToLower() }
  $newEntries[$rel] = $hash
}

# Merge and sort
$merged = @{}
foreach ($kv in $existing.GetEnumerator()) { $merged[$kv.Key] = $kv.Value }
foreach ($kv in $newEntries.GetEnumerator()) { $merged[$kv.Key] = $kv.Value }

$lines = $merged.Keys | Sort-Object | ForEach-Object { Format-IndexLine $merged[$_] $_ }

# Write index
Set-Content -LiteralPath $indexPath -Value $lines -NoNewline:$false -Encoding UTF8
Write-Info ("Wrote index: {0}" -f (Split-Path -Leaf $indexPath))

# Write master hash for the index itself
$indexHash = (Get-FileHash -Algorithm $Algorithm -LiteralPath $indexPath).Hash.ToLower()
$masterSidecar = "$indexPath.$($Algorithm.ToLower())"
Set-Content -LiteralPath $masterSidecar -Value $indexHash -NoNewline:$true -Encoding ASCII
Write-Info ("Master index hash: $indexHash -> {0}" -f (Split-Path -Leaf $masterSidecar))

Write-Info "Done"

