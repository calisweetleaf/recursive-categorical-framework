# Recursive Categorical Framework – Provenance Dossier

## Identity & Contacts
- Legal name: Christian “Trey” Rowell  
- Research alias: Daeron Blackfyre  
- Primary email: treyrowell1826@gmail.com  
- Secondary email: daeronblackfyre18@gmail.com  
- ORCID: 0009-0008-6550-6316 — https://orcid.org/0009-0008-6550-6316  

## Core References
- DOI: https://doi.org/10.5281/zenodo.17567903  
- Academia mirror: https://www.academia.edu/144895498/Recursive_Categorical_Framework_RCF_A_Novel_Theoretical_Foundation_for_Synthetic_Consciousness?source=swp_share  
- Repository: https://github.com/daeron-bf/recursive-categorical-framework  

## Publication Timeline
| Date (UTC) | Channel | Notes |
| --- | --- | --- |
| 2025-11-10 | Academia.edu | Pre-doi release referencing manuscript PDF |
| 2025-11-11 | Zenodo | DOI minted, ORCID-linked import, metadata in 11-11-202-zenodo.json |
| 2025-11-11 | Repository | Hash ledger initialized (SHA256SUMS + signature) |
| Pending | OSF / MetaArXiv | Submission queued via ORCID |
| Pending | Perplexity / Gemini | Planned mirrors per publication manifest |

## Integrity Artifacts
- `hash-index.ps1` — deterministic SHA hashing script (included in repo).  
- `SHA256SUMS` — current digest index for all tracked files.  
- `SHA256SUMS.sha256` — hash of the index itself.  
- `SHA256SUMS.asc` — detached PGP signature (fingerprint `6E68BB678FD9630755B70D621A4CD5DF1B0A33AE`).  
- `allowed_signers.yaml` — trusted signer declaration.  
- `CITATION.cff` — machine-readable citation metadata.  
- `provenence/orcid-works.pdf` — ORCID export confirming DOI linkage.  
- `provenence/zenodo-doi.txt` — DOI text reference.  

## Verification Procedure
1. Import public key:  
   ```bash
   gpg --armor --export 6E68BB678FD9630755B70D621A4CD5DF1B0A33AE > treyrowell_pubkey.asc
   gpg --import treyrowell_pubkey.asc
   ```
2. Verify integrity list:  
   ```bash
   gpg --verify SHA256SUMS.asc SHA256SUMS
   sha256sum --check SHA256SUMS
   ```
3. Cross-check DOI + ORCID records against repo commit hash when mirroring.

Maintain this dossier with each release to keep provenance synchronized across Zenodo, ORCID, and Git.
