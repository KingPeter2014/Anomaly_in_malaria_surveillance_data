# Repository Transition for ZinnyWorld

You asked to **change the repository before coding**.

This project currently contains a planning scaffold at `zinnyworld/`, but implementation should happen in a **separate repository** named `zinnyworld`.

## What was added
- `tools/create_zinnyworld_repo.sh` to bootstrap a standalone local repository from the existing planning assets.

## How to create the new local repo
```bash
bash tools/create_zinnyworld_repo.sh ../zinnyworld
```

## After local creation
1. Create a new empty remote repository (e.g., GitHub) named `zinnyworld`.
2. Connect and push:
```bash
cd ../zinnyworld
git remote add origin <your_repo_url>
git push -u origin main
```

## Why this step first
This ensures all upcoming PHP/MySQL coding occurs in the correct project repository and keeps this current malaria repository clean from application implementation work.
