# cudabsgs

CUDA-accelerated BSGS build of `keyhunt` for secp256k1 public-key puzzle searches.

This repository is a source-only distribution. Build products, bloom files, BSGS tables, CUDA native caches, logs, and found-key files are intentionally ignored.

## Status

The CUDA path accelerates the BSGS bloom/table probing workflow while keeping the original keyhunt command-line interface. It has been validated with a known-key BSGS test in `tests/find_known_53.txt`.

This is performance research software. Use it only on ranges and targets you are authorized to test.

## Requirements

Linux or WSL is recommended.

Required:

```bash
sudo apt update
sudo apt install -y git build-essential
```

For CUDA builds:

```bash
nvcc --version
nvidia-smi
```

You need an NVIDIA driver and CUDA toolkit that support your GPU architecture.

## Clone

```bash
git clone https://github.com/kemo159/cudabsgs.git
cd cudabsgs
```

## Build

CPU build:

```bash
make clean
make -j$(nproc)
```

CUDA build:

```bash
make clean
make -j$(nproc) keyhunt_cuda CUDA_ARCH=sm_120
```

Native Windows CUDA build:

```cmd
set CUDA_ARCH=sm_120
build_win.bat
```

The Windows build requires Visual Studio 2022 Build Tools and the NVIDIA CUDA toolkit. `build_win.bat` calls `vcvars64.bat` automatically and writes `keyhunt_cuda.exe` in the repository root.

Set `CUDA_ARCH` for your GPU:

| GPU family | Example GPUs | CUDA_ARCH |
|---|---|---|
| Blackwell | RTX 5080, RTX 5090 | `sm_120` |
| Ada | RTX 4090, RTX 6000 Ada, L40S | `sm_89` |
| Ampere | RTX 3090, A4000, A5000, A6000 | `sm_86` |
| Ampere datacenter | A100 | `sm_80` |
| Turing | RTX 8000 | `sm_75` |

Examples:

```bash
make -j$(nproc) keyhunt_cuda CUDA_ARCH=sm_86
make -j$(nproc) keyhunt_cuda CUDA_ARCH=sm_89
make -j$(nproc) keyhunt_cuda CUDA_ARCH=sm_120
```

## Run

Example BSGS run:

```bash
./keyhunt_cuda -m bsgs -f tests/130.txt -k 512 -b 130 -S -t 32
```

Larger GPU tables can improve speed if they fit in VRAM:

```bash
./keyhunt_cuda -m bsgs -f tests/130.txt -k 1024 -b 130 -S -t 32
./keyhunt_cuda -m bsgs -f tests/130.txt -k 2048 -b 130 -S -t 32
```

Watch GPU usage:

```bash
watch -n 1 nvidia-smi
```

## Known-Key Validation

Build the CUDA binary, then run:

```bash
./keyhunt_cuda -m bsgs -f tests/find_known_53.txt -r 10000000000000:1fffffffffffff -S -t 32 -k 128
```

Expected result:

```text
Key found privkey 180788e47e326c
Publickey 020faaf5f3afe58300a335874c80681cf66933e2a7aeb28387c0d28bb048bc6349
```

The found key is also written to:

```text
KEYFOUNDKEYFOUND.txt
```

## Tuning

`-k` controls BSGS table size. Bigger values use more memory and can increase throughput if the GPU has enough VRAM.

`-t` controls CPU worker threads, not CUDA threads. More CPU threads can help feed the GPU, but too many can reduce speed due to scheduling, cache, and synchronization overhead.

Suggested thread sweep:

```bash
for t in 16 24 32 48 64 96 128; do
  ./keyhunt_cuda -m bsgs -f tests/130.txt -k 512 -b 130 -S -t $t
done
```

Compare the later total-speed lines, not the first instant line.

Suggested 32 GB VRAM `-k` sweep:

```bash
./keyhunt_cuda -m bsgs -f tests/130.txt -k 1024 -b 130 -S -t 32
./keyhunt_cuda -m bsgs -f tests/130.txt -k 1536 -b 130 -S -t 32
./keyhunt_cuda -m bsgs -f tests/130.txt -k 2048 -b 130 -S -t 32
```

## Cache Files

The first run for a given `-k` may generate:

```text
keyhunt_bsgs_*.blm
keyhunt_bsgs_*.tbl
keyhunt_bsgs_cuda_*.gcache
```

Later runs can reuse these files and start faster. They are ignored by git because they are generated artifacts and can be very large.

Cache files are not guaranteed to be portable between Linux/WSL and native Windows builds. If a native Windows run reports a checksum mismatch on cache files created by Linux/WSL, delete the matching `keyhunt_bsgs_*` and `keyhunt_bsgs_cuda_*` files and let the Windows build regenerate them.

## Other Targets

CUDA selftests:

```bash
make cuda_selftest CUDA_ARCH=sm_120
./cuda/secp256k1_cuda_selftest
./cuda/bsgs_cuda_selftest
```

CPU helper builds:

```bash
make bsgsd
make legacy
```

Clean generated binaries and objects:

```bash
make clean
```

## Notes

Multi-GPU is not a single shared VRAM pool. Each GPU normally needs its own table/cache copy and should search a different range slice.

This fork is based on AlbertoBSD's keyhunt project and keeps the original license in `LICENSE`.
