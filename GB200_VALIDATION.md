# GB200 Validation - Instructions

Hey! Thanks for helping with the GB200 validation. This should take about 10-15 minutes total (mostly build time).

## What You'll Do

1. Download 1 file (Dockerfile)
2. Build a Docker container (~5-10 min - handles all setup automatically)
3. Run validation (~5 min - tests the kernels)
4. Send me 2 files back

## Commands (Copy-Paste These)

```bash
# 1. Create working directory
mkdir ~/gb200_validation
cd ~/gb200_validation

# 2. Download the Dockerfile
wget https://raw.githubusercontent.com/eous/scratchpad/main/Dockerfile.gb200-validation

# 3. Build the Docker image (installs CUDA, PyTorch, FlashMLA, DeepGEMM - takes ~5-10 minutes)
docker build -f Dockerfile.gb200-validation -t gb200-validation .

# 4. Run the validation (tests all kernels - takes ~5 minutes)
mkdir output
docker run --gpus all -v $(pwd)/output:/output gb200-validation

# You'll see output like:
# ================================================================================
# SM100 Reference Output Generator
# ================================================================================
# [1/6] Collecting system information...
#   GPU: NVIDIA B200
#   Compute Capability: (10, 0)
# [2/6] Testing FlashMLA Decode kernels...
#     ✓ small_single: 0.000234 mean, 1.23ms
#     ✓ medium_multi: 0.000312 mean, 2.45ms
#     ... (30+ more tests)
#
# ✅ GB200 Validation Complete!
# Output files:
# -rw-r--r-- 1 root root  15M sm100_reference.pkl
# -rw-r--r-- 1 root root  78K sm100_reference.json
# -rw-r--r-- 1 root root 7.2M sm100_validation_results.tar.gz
# -rw-r--r-- 1 root root  281 checksums.txt
```

## Send Me These 2 Files

```bash
cd ~/gb200_validation/output

# These are the files I need:
ls -lh sm100_validation_results.tar.gz  # ~7-15 MB compressed
ls -lh checksums.txt                     # ~300 bytes

# Transfer via scp, email, Slack, cloud storage, or whatever works for you
```

## That's It!

The Docker container handles:
- ✅ CUDA environment setup
- ✅ Building FlashMLA
- ✅ Building DeepGEMM
- ✅ Running validation tests
- ✅ Creating compressed archive
- ✅ Generating checksums

You just run 3 commands and send me 2 files.

---

## Troubleshooting

**"docker: command not found"**
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

**"nvidia-container-toolkit not found"**
```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Build takes too long / running out of time**

Use quick mode (30 seconds instead of 5 minutes):
```bash
docker run --gpus all -v $(pwd)/output:/output gb200-validation \
  python3 generate_sm100_reference.py --output /output/sm100_reference.pkl --quick
```

---

**Questions? Just let me know!**
