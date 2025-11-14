param(
  [string]$Path = ".",
  [ValidateSet('SHA256','SHA1','SHA384','SHA512')]
  [string]$Algorithm = 'SHA256',
  [string]$IndexFile = 'SHA256SUMS.txt',
  [switch]$SkipExisting,          # Skip files already listed in the index
  [switch]$UseSidecar,            # Reuse sidecar hashes like file.ext.sha256 / .sha1
  [switch]$Verify,                # Verify files against the index instead of writing
  [switch]$SignWithGPG,           # Auto-sign the index with GPG
  [switch]$TimestampProof,        # Generate OpenTimestamps proof (requires ots-cli)
  [string]$ManifestVersion = "1.0", # Track manifest format version
  [string]$ReleaseVersion = "",   # Release version (e.g., "v1.0.0")
  [string]$ReleaseTitle = "",     # Release title for metadata
  [string[]]$Exclude = @('.git', '.venv', 'node_modules', "$IndexFile", "$IndexFile.sha256")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info([string]$msg) { Write-Host "[hash-index] $msg" -ForegroundColor Cyan }
function Write-Success([string]$msg) { Write-Host "[hash-index] $msg" -ForegroundColor Green }
function Write-Warn([string]$msg) { Write-Host "[hash-index] $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg) { Write-Host "[hash-index] $msg" -ForegroundColor Red }

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
    if (-not $line -or $line.StartsWith('#')) { continue }
    $m = [regex]::Match($line, "^([A-Fa-f0-9]{$len})\s\s(.*)$")
    if ($m.Success) {
      $h = $m.Groups[1].Value.ToLower()
      $p = $m.Groups[2].Value
      $map[$p] = $h
    }
  }
  return $map
}

function Format-IndexLine([string]$hash, [string]$relPath) {
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
    if ($rel -like "$pat*" -or ([System.IO.Path]::GetFileName($rel) -like $pat)) { return $true }
  }
  return $false
}

function Get-GitCommit {
  if (Test-Path .git) {
    $commit = git rev-parse HEAD 2>$null
    if ($LASTEXITCODE -eq 0) { return $commit }
  }
  return "N/A"
}

function Get-GitTag {
  if (Test-Path .git) {
    $tag = git describe --tags --exact-match 2>$null
    if ($LASTEXITCODE -eq 0) { return $tag }
  }
  return "N/A"
}

function Add-ManifestHeader([string]$path, [string]$alg) {
  $timestamp = (Get-Date).ToUniversalTime().ToString("o")
  $gitCommit = Get-GitCommit
  $gitTag = Get-GitTag
  
  $header = @"
# Hash Manifest v$ManifestVersion
# Algorithm: $alg
# Generated: $timestamp
# Git Commit: $gitCommit
# Git Tag: $gitTag
"@

  if ($ReleaseVersion) {
    $header += "`n# Release Version: $ReleaseVersion"
  }
  if ($ReleaseTitle) {
    $header += "`n# Release Title: $ReleaseTitle"
  }
  
  $header += "`n# Tool: hash-index.ps1`n#`n"
  
  $content = Get-Content -LiteralPath $path -Raw
  Set-Content -LiteralPath $path -Value "$header$content" -NoNewline:$false -Encoding UTF8
}

function Test-GPGAvailable {
  try {
    $null = Get-Command gpg -ErrorAction Stop
    return $true
  } catch {
    return $false
  }
}

function Test-OTSAvailable {
  try {
    $null = Get-Command ots -ErrorAction Stop
    return $true
  } catch {
    return $false
  }
}

function Sign-WithGPG([string]$filePath) {
  if (-not (Test-GPGAvailable)) {
    Write-Warn "GPG not found. Skipping signature."
    return $false
  }
  
  Write-Info "Signing with GPG..."
  $sigPath = "$filePath.asc"
  
  # Remove existing signature
  if (Test-Path -LiteralPath $sigPath) {
    Remove-Item -LiteralPath $sigPath -Force
  }
  
  & gpg --detach-sign --armor --output $sigPath $filePath
  
  if ($LASTEXITCODE -eq 0) {
    Write-Success "GPG signature: $(Split-Path -Leaf $sigPath)"
    return $true
  } else {
    Write-Err "GPG signing failed"
    return $false
  }
}

function Create-Timestamp([string]$filePath) {
  if (-not (Test-OTSAvailable)) {
    Write-Warn "OpenTimestamps (ots) not found. Skipping timestamp proof."
    Write-Info "Install with: pip install opentimestamps-client"
    return $false
  }
  
  Write-Info "Creating OpenTimestamps proof..."
  $otsPath = "$filePath.ots"
  
  # Remove existing timestamp
  if (Test-Path -LiteralPath $otsPath) {
    Remove-Item -LiteralPath $otsPath -Force
  }
  
  & ots stamp $filePath
  
  if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $otsPath)) {
    Write-Success "Timestamp proof: $(Split-Path -Leaf $otsPath)"
    Write-Info "Verify later with: ots verify $otsPath"
    return $true
  } else {
    Write-Warn "Timestamp creation failed (this is normal if not connected to Bitcoin node)"
    return $false
  }
}

function Create-ProvenanceFile([string]$root, [string]$indexPath) {
  $provPath = Join-Path $root "PROVENANCE.yaml"
  $timestamp = (Get-Date).ToUniversalTime().ToString("o")
  $commit = Get-GitCommit
  $tag = Get-GitTag
  
  # Get GPG key fingerprint if available
  $gpgKey = "N/A"
  if (Test-GPGAvailable) {
    $keyInfo = gpg --list-secret-keys --keyid-format LONG 2>$null
    if ($LASTEXITCODE -eq 0 -and $keyInfo) {
      $match = [regex]::Match($keyInfo, 'sec\s+\S+/([A-F0-9]+)')
      if ($match.Success) {
        $gpgKey = $match.Groups[1].Value
      }
    }
  }
  
  $provContent = @"
disclosure:
  title: "$ReleaseTitle"
  version: "$ReleaseVersion"
  release_date: "$timestamp"
  author: "Christian Trey Rowell"
  
chain:
  - event: "Hash manifest generation"
    date: "$timestamp"
    git_commit: "$commit"
    git_tag: "$tag"
    
signatures:
  gpg_key: "$gpgKey"
  index_files:
    - $(Split-Path -Leaf $indexPath)
    - $(Split-Path -Leaf $indexPath).asc
    - $(Split-Path -Leaf $indexPath).ots
    
verification:
  instructions: |
    1. Verify GPG signature:
       gpg --verify $(Split-Path -Leaf $indexPath).asc $(Split-Path -Leaf $indexPath)
    
    2. Verify file integrity:
       sha256sum -c $(Split-Path -Leaf $indexPath)
    
    3. Verify timestamp (after blockchain confirmation):
       ots verify $(Split-Path -Leaf $indexPath).ots
"@

  Set-Content -LiteralPath $provPath -Value $provContent -Encoding UTF8
  Write-Success "Provenance file: $(Split-Path -Leaf $provPath)"
}

# Main execution
$root = Resolve-Path -LiteralPath $Path
$indexPath = Join-Path -Path $root -ChildPath $IndexFile

# Adjust index filename based on algorithm if not explicitly set
if ($PSBoundParameters.ContainsKey('Algorithm') -and -not $PSBoundParameters.ContainsKey('IndexFile')) {
  switch ($Algorithm) {
    'SHA1'   { $indexPath = Join-Path $root 'SHA1SUMS.txt' }
    'SHA384' { $indexPath = Join-Path $root 'SHA384SUMS.txt' }
    'SHA512' { $indexPath = Join-Path $root 'SHA512SUMS.txt' }
    default  { }
  }
}

# Update exclude list to include signature and timestamp files
$Exclude += "$IndexFile.asc"
$Exclude += "$IndexFile.ots"
$Exclude += "PROVENANCE.yaml"

# Load existing index entries
$existing = Parse-Index $indexPath $Algorithm

if ($Verify) {
  if (-not (Test-Path -LiteralPath $indexPath)) { throw "Index file not found: $indexPath" }
  Write-Info "Verifying files listed in $(Split-Path -Leaf $indexPath)"
  
  # Verify GPG signature if available
  $sigPath = "$indexPath.asc"
  if (Test-Path -LiteralPath $sigPath) {
    if (Test-GPGAvailable) {
      Write-Info "Verifying GPG signature..."
      & gpg --verify $sigPath $indexPath
      if ($LASTEXITCODE -eq 0) {
        Write-Success "GPG signature valid"
      } else {
        Write-Err "GPG signature verification failed"
      }
    } else {
      Write-Warn "GPG signature present but GPG not available"
    }
  }
  
  # Verify file hashes
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
  if ($fail -gt 0) { throw "$fail file(s) failed verification" } else { Write-Success "Verification succeeded" }
  return
}

Write-Info "Scanning '$root' for files to hash ($Algorithm)"

# Build set of files to hash
$all = Get-ChildItem -LiteralPath $root -Recurse -File -Force
$toHash = @()
foreach ($f in $all) {
  if (Should-Exclude $root $f.FullName $Exclude) { continue }
  # Never hash the index or its signature files
  if ($f.FullName -ieq $indexPath) { continue }
  if ($f.FullName -ieq "$indexPath.$($Algorithm.ToLower())") { continue }
  if ($f.FullName -ieq "$indexPath.asc") { continue }
  if ($f.FullName -ieq "$indexPath.ots") { continue }
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
  if (-not $hash) { 
    $hash = (Get-FileHash -Algorithm $Algorithm -LiteralPath $f.FullName).Hash.ToLower() 
  }
  $newEntries[$rel] = $hash
}

# Merge and sort
$merged = @{}
foreach ($kv in $existing.GetEnumerator()) { $merged[$kv.Key] = $kv.Value }
foreach ($kv in $newEntries.GetEnumerator()) { $merged[$kv.Key] = $kv.Value }

$lines = $merged.Keys | Sort-Object | ForEach-Object { Format-IndexLine $merged[$_] $_ }

# Write index
Set-Content -LiteralPath $indexPath -Value $lines -NoNewline:$false -Encoding UTF8

# Add metadata header
Add-ManifestHeader $indexPath $Algorithm

Write-Success ("Wrote index: {0} ({1} files)" -f (Split-Path -Leaf $indexPath), $merged.Count)

# Write master hash for the index itself
$indexHash = (Get-FileHash -Algorithm $Algorithm -LiteralPath $indexPath).Hash.ToLower()
$masterSidecar = "$indexPath.$($Algorithm.ToLower())"
Set-Content -LiteralPath $masterSidecar -Value $indexHash -NoNewline:$true -Encoding ASCII
Write-Success ("Master index hash: $indexHash")

# Sign with GPG if requested
if ($SignWithGPG) {
  Sign-WithGPG $indexPath
}

# Create timestamp proof if requested
if ($TimestampProof) {
  Create-Timestamp $indexPath
}

# Create provenance file if release info provided
if ($ReleaseVersion -or $ReleaseTitle) {
  Create-ProvenanceFile $root $indexPath
}

Write-Success "Done"
Write-Info ""
Write-Info "Next steps:"
Write-Info "  1. Review generated files"
if ($SignWithGPG -and (Test-Path "$indexPath.asc")) {
  Write-Info "  2. Commit and sign: git commit -S -m 'Release $ReleaseVersion'"
  Write-Info "  3. Create signed tag: git tag -s $ReleaseVersion -m 'Release $ReleaseVersion'"
} else {
  Write-Info "  2. Consider signing: .\hash-index.ps1 -SignWithGPG"
}
Write-Info "  4. Push with tags: git push origin main --tags"