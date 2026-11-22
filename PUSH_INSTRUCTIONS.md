# How to Push This Folder to GitHub

This folder (`github_upload/`) contains the complete, clean, self-contained open-source release of the **SBP AI API Security Mesh** engineered for State Bank of Pakistan (SBP) Regulatory Sandbox Cohort 2 evaluators.

---

## Option A: New Repository (Recommended)

1. Open your browser and go to [https://github.com/new](https://github.com/new).
2. **Repository name**: `sbp-api-security-mesh`
3. **Description**: `Real-time AI Anomaly Detection & Zero-Trust Reverse-Proxy Mesh for Pakistan Open Banking (SBP Regulatory Sandbox Cohort 2)`
4. **Visibility**: Select **Public** *(Required: SBP evaluators must access without credentials)*.
5. **Initialize this repository with**: **Leave unchecked** (do NOT add README, .gitignore, or license — they are already configured in this directory).
6. Click **Create repository**.
7. In your terminal, run the following commands:

```bash
cd /home/ahmad/Desktop/SBP_PROJECT/github_upload
git init
git branch -M main
git add .
git commit -m "Initial commit: SBP AI API Security Mesh for Cohort 2"
git remote add origin https://github.com/ba456jutt-cloud/zero_trust_api_mesh.git
git push -u origin main
```

*(Replace `YOUR_USERNAME` with your actual GitHub username).*

---

## Option B: Existing Repository

If you already created a repository on GitHub:

```bash
cd /home/ahmad/Desktop/SBP_PROJECT/github_upload
git init
git branch -M main
git add .
git commit -m "SBP Sandbox Cohort 2 submission release"
git remote add origin https://github.com/YOUR_USERNAME/EXISTING_REPO.git
git push -u origin main
```

---

## Pre-Push Verification Checks

### 1. Secret Leak Check
Run this check to confirm zero credentials, tokens, or private keys are exposed:
```bash
grep -rE "password|secret|api_key|token.*=.*['\"]" . \
    --include="*.py" --include="*.md" --include="*.json" \
    | grep -v "test_\|example\|placeholder" | head -20
```

### 2. Disk Size Check
GitHub recommends repositories remain below 1 GB. This upload package is $\sim 5.9\text{ MB}$:
```bash
du -sh .
```

### 3. Verify No Model Binaries
Ensure heavy `.joblib` and `.pt` files are excluded (evaluators recreate them cleanly via `./scripts/reproduce_demo.sh`):
```bash
find . -name "*.joblib" -o -name "*.pt" -o -name "*.pkl"
# (Output should be empty)
```

---

## Post-Push Verification Checklist

- [ ] Visit `https://github.com/ba456jutt-cloud/zero_trust_api_mesh` in an incognito / private browser window to confirm it is publicly accessible without login.
- [ ] Verify that `data/external/apisentry/dataset_processed.csv` is present in the repository file tree.
- [ ] Click green **Code $\to$ Download ZIP** to verify direct download functionality.
- [ ] Note the latest commit hash (e.g. from `git rev-parse HEAD`).
- [ ] Update your repository URL and commit hash in your formal application documents:
  - `SBP_APPLICATION/05_attachments_manifest.md` (Item #15)
  - `SBP_APPLICATION/06_email_draft.md`

