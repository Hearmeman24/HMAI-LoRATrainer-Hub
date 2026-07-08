---
name: hmai-lora-trainer
description: Deploy the HMAI LoRA Trainer serverless endpoint on RunPod and train LoRAs end-to-end on it — SDXL, Wan 2.2, Qwen-Image (+2512), FLUX.2 Klein 9B, Krea 2 (+Turbo), Z-Image Turbo, and Ideogram 4. Use this skill whenever the user wants to train a LoRA, fine-tune a diffusion model on their own images, deploy the aio-lora-trainer template, set up the HMAI trainer endpoint, or asks anything about dataset prep, config overrides, checkpoints, or retrieving trained .safetensors from this endpoint — even if they just say "train a LoRA of X" without naming the service.
---

# HMAI LoRA Trainer — deploy & train end-to-end

Serverless LoRA training on RunPod. You send a zip of images + captions and a
trigger word; the endpoint trains and uploads `.safetensors` checkpoints to the
user's S3 or Cloudflare R2 bucket.

This skill has two halves: **Part 1 — deploy the endpoint** (one-time setup) and
**Part 2 — train a LoRA** (the repeatable workflow). If the user already has an
endpoint ID, skip to Part 2.

---

## Part 1 — Deploy the endpoint

### Option A: RunPod Hub (easiest)

Find **HMAI LoRA Trainer** on the RunPod Hub and click **Deploy**. The template
is pre-configured; the user only picks a GPU and fills in environment variables.

### Option B: Manual serverless endpoint

Create a new RunPod Serverless endpoint with:

| Setting | Value |
|---|---|
| Docker image | `hearmeman/aio-lora-trainer:v3.0.1` |
| GPU | **80 GB VRAM recommended** (H100, H200, A100 80GB). Image-only jobs can run on less (e.g. 48 GB), but 80 GB is the safe default — video (Wan 2.2) and the larger image models need it. |
| Container disk | **60 GB minimum** — base model weights are large. |
| Execution timeout | **Raise it.** Training runs for hours; RunPod's default execution timeout will kill the job mid-training. Set it to the max you expect a run to take (the worker itself enforces a 12-hour cap via `MAX_TRAINING_HOURS`). |
| Network volume | Optional but strongly recommended. Base models are cached under `/runpod-volume/models` — with a volume, the second job skips a 20–60 GB model download; without one, every cold worker re-downloads from scratch. |

### Environment variables — S3/R2 is NOT optional in practice

**Set up S3 or R2 before training anything real.** The worker uploads finished
checkpoints to the bucket and returns presigned download URLs. Without storage
configured, the job still trains — but the output response says
`"storage": "local_only"` and the `.safetensors` files exist only on the
serverless worker's ephemeral disk. When that worker scales down (minutes after
the job ends), **the files are gone. They go nowhere. Hours of GPU time,
nothing to download.** Treat missing storage credentials as a setup error and
tell the user so before submitting a training job.

Set ONE of these two groups (R2 wins if both are set):

| Storage | Variables |
|---|---|
| AWS S3 | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET`, `S3_REGION` (default `us-east-1`) |
| Cloudflare R2 | `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` |

Plus, when needed:

| Variable | When |
|---|---|
| `HF_TOKEN` | Required for gated base models: **FLUX.2 Klein 9B** and **Ideogram 4**. The user must also accept each model's license on Hugging Face with that account. Jobs on these models fail with a 401/403 download error without it. |
| `CIVITAI_API_KEY` | Only for SDXL training on a custom CivitAI base checkpoint (`civitai_model_id`). |

### Verify the deployment (smoke test)

Before any real job, confirm the endpoint is alive — this validates the payload
and resolves the model without downloading or training, and returns in seconds:

```bash
curl -s -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"smoke": true, "model_type": "sdxl", "dataset_zip_url": "x", "trigger_word": "x"}}'
```

Expect `"ok": true, "smoke": true` in the output.

---

## Part 2 — Train a LoRA (agent workflow)

Follow these steps in order.

### Step 1 — Pick the model

| `model_type` | Model | Notes |
|---|---|---|
| `sdxl` | Stable Diffusion XL | Optional `civitai_model_id` to train on a CivitAI checkpoint. Default res 1024. |
| `wan2.2` | Wan 2.2 T2V A14B (video) | **Requires `noise_variant`: `"high"` or `"low"` — one per job.** A complete Wan 2.2 LoRA needs BOTH variants, so submit **two jobs** (one high, one low). Accepts video files in the dataset. |
| `qwen_image` | Qwen-Image | Trains at resolution 1328. |
| `qwen_image_2512` | Qwen-Image 2512 | Trains at resolution 1328. |
| `z_image` | Z-Image Turbo | Turbo training adapter applied automatically. |
| `ideogram4` | Ideogram 4 | **Gated — needs `HF_TOKEN`.** |
| `flux_klein_9b` | FLUX.2 Klein 9B | **Gated — needs `HF_TOKEN`.** |
| `krea2` | Krea 2 (raw base) | |
| `krea2_turbo` | Krea 2 Turbo | Turbo training adapter applied automatically. |

Training defaults for every model: rank 32, lr 1e-4, adamw8bit, 100 epochs,
checkpoint every 5 epochs. Override via `config_overrides` (Step 4).

### Step 2 — Prepare the dataset

A flat **zip** of media files with a matching `.txt` caption per file
(same basename):

```
my_dataset.zip
├── 01.png
├── 01.txt      ← caption text, should mention the trigger word
├── 02.jpg
├── 02.txt
└── ...
```

- Images: `.png` `.jpg` `.jpeg` `.webp`. Videos (Wan 2.2): `.mp4` `.mov` `.avi` `.mkv`.
- Every media file needs a caption; uncaptioned files are warned about and can
  degrade results. Include the trigger word in each caption.
- A zip that wraps everything in one folder is fine (it's unwrapped
  automatically); `__MACOSX` junk is ignored.
- 15–50 well-captioned images is a solid range for a character/style LoRA.

Upload the zip anywhere the endpoint can GET it — a presigned S3/R2 URL or any
direct public link — and keep that URL. Redirects are followed; the URL must
return the raw zip, not an HTML page (Google Drive share links don't work).

### Step 3 — Submit the job

Training takes hours: always use the async `/run` endpoint (never `/runsync`).

```bash
curl -s -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
        "input": {
          "model_type": "krea2",
          "dataset_zip_url": "https://your-storage/my_dataset.zip?X-Amz-...",
          "trigger_word": "mystyle"
        }
      }'
```

Save the returned job `id`. For Wan 2.2, submit two of these — identical except
`"noise_variant": "high"` and `"noise_variant": "low"`.

### Step 4 — (Optional) config_overrides

Common overrides, passed as `"config_overrides": {...}` in the input:

| Key | Meaning |
|---|---|
| `epochs` | Total training epochs (default 100) |
| `save_every_n_epochs` | Checkpoint interval (default 5) |
| `adapter.rank` | LoRA rank (default 32; alpha is set equal to rank) |
| `optimizer.lr` | Learning rate (default 1e-4) |
| `optimizer.type` | Optimizer (default `adamw8bit`) |
| `dataset.num_repeats` | Repeats per image per epoch |
| `dataset.resolutions` | e.g. `[1024]` — training resolution list |
| `micro_batch_size_per_gpu` | Batch size |
| `gradient_accumulation_steps` | Gradient accumulation |
| `warmup_steps` | LR warmup steps |

Unknown or unsupported keys are **silently ignored** (logged server-side, never
an error) — so double-check spelling; a typo'd override just doesn't take effect.

### Step 5 — Poll until done

```bash
curl -s https://api.runpod.ai/v2/<ENDPOINT_ID>/status/<JOB_ID> \
  -H "Authorization: Bearer <RUNPOD_API_KEY>"
```

Poll every 1–5 minutes. `IN_QUEUE` → `IN_PROGRESS` → `COMPLETED` (or `FAILED`).
Expect the first run on a cold worker to spend significant time downloading the
base model before training starts. Checkpoints are uploaded to the bucket
incrementally as training progresses, so partial results survive even a
mid-run failure.

### Step 6 — Retrieve the LoRA

On `COMPLETED`, the `output` looks like:

```json
{
  "ok": true,
  "model_type": "krea2",
  "trigger_word": "mystyle",
  "output_files": [
    {"filename": "mystyle_krea2_epoch5.safetensors", "url": "https://..."},
    {"filename": "mystyle_krea2_epoch10.safetensors", "url": "https://..."}
  ],
  "presigned_urls": ["https://...", "https://..."],
  "timing": {"dataset_download_s": 12.3, "model_download_s": 340.1, "training_s": 7211.0, "total_s": 7602.4}
}
```

- One file per saved epoch, named `<trigger>_<model>[_variant]_epoch<N>.safetensors`.
  The **highest epoch is not automatically the best** — later checkpoints can be
  overtrained. Suggest the user test a few (e.g. epoch 50 vs 75 vs 100).
- `presigned_urls` **expire after 7 days** — download what's needed promptly;
  the objects themselves stay in the bucket under `lora-outputs/<job_id>/`.
- If the output contains `"storage": "local_only"`, storage wasn't configured
  and the files are stranded on the ephemeral worker — see Part 1. Fix the env
  vars and re-run.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `error_type: "VALIDATION"` | Bad payload — the `error` message says which field. Common: missing `noise_variant` on wan2.2, `noise_variant`/`civitai_model_id` sent for a model that doesn't take it. |
| 401/403 during model download | Gated model (FLUX.2, Ideogram 4) without `HF_TOKEN`, or license not accepted on HF. |
| CUDA out of memory | GPU too small for the model — redeploy with 80 GB VRAM. |
| Job killed before finishing | Endpoint execution timeout too low — raise it in endpoint settings. Checkpoints already uploaded are safe. |
| `"storage": "local_only"` in output | No S3/R2 env vars — files unreachable once the worker stops. Configure storage, re-run. |
| Dataset error "No media files found" | Zip is empty, wrongly nested, or uses unsupported extensions. |
| Override didn't change anything | Key was misspelled/unsupported — overrides never error, they're ignored. Check the table in Step 4. |
| Very slow start | Cold worker downloading base model weights (tens of GB). Attach a network volume to cache them across jobs. |
