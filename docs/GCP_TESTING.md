# Testing DeciGrep on GCP (Compute Engine with NVIDIA L4)

Follow these instructions to provision a Google Cloud Compute Engine instance
with an NVIDIA L4 GPU (24 GB VRAM) and run DeciGrep's test suite and live
smoke tests with Ollama.

---

## Prerequisites

- A Google Cloud project with billing enabled.
- [Google Cloud CLI (`gcloud`)](https://cloud.google.com/sdk/docs/install)
  installed and authenticated:

  ```bash
  gcloud auth login
  gcloud config set project <PROJECT_ID>
  ```

- **GPU quota:** ensure your project has quota for `NVIDIA_L4_GPUS` in your
  target region (e.g., `us-central1`). How to increase quotas: https://cloud.google.com/compute/quotas

## 1. Provision the G2 instance

Create a `g2-standard-4` instance (1× NVIDIA L4 GPU, 4 vCPUs, 16 GB RAM):

```bash
gcloud compute instances create ollama-test-vm \
    --zone=us-central1-a \
    --machine-type=g2-standard-4 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=common-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-balanced \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT
```

> **Tip:** add `--provisioning-model=SPOT` to reduce costs for temporary
> testing (subject to preemption).

> **Note:** using Google's Deep Learning VM image
> (`common-cu121-...`) with `--metadata="install-nvidia-driver=True"`
> automatically installs the compatible NVIDIA drivers and CUDA toolkit on
> boot.

## 2. Connect to the VM and verify the GPU

```bash
gcloud compute ssh ollama-test-vm --zone=us-central1-a
```

Verify that the GPU and drivers are recognized:

```bash
nvidia-smi
```

You should see the NVIDIA L4 GPU listed with ~24 GB of VRAM.

## 3. Install system dependencies and Python

```bash
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv
```

## 4. Install and start Ollama

DeciGrep needs **Ollama v0.35.0 or later** (System One API). Install it:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify Ollama is running and has GPU access:

```bash
ollama --version
```

Make sure the Ollama server is actually up. The install script usually
enables and starts a `systemd` service — check it with
`systemctl status ollama`. If the service is not running (or on a system
without systemd), start the server manually:

```bash
ollama serve
```

To expose the server to other machines (e.g. your laptop running DeciGrep
against this VM's GPU), bind it to all interfaces:

```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Pull the decision model used by DeciGrep (`nimble`). Note: this replaces the
`llama3:8b` example from generic Ollama setups — DeciGrep requires a
*decision* model:

```bash
ollama pull nimble
```

Confirm the model is available:

```bash
ollama list
```

To verify the model is offloaded to VRAM, run `nvidia-smi` in another
terminal while making a request, or check `ollama ps`.

## 5. Clone the repository and set up the utility

Authenticate with GitHub:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
```

Copy the public key output and add it to your
[GitHub SSH keys](https://github.com/settings/keys).

Clone the repository:

```bash
git clone git@github.com:<your-org>/<your-repo>.git
cd <your-repo>
```

Set up the Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the unit test suite (either runner works; the tests are written with
`unittest`):

```bash
python -m unittest discover -s tests -v
```

Run a live smoke test against the locally running Ollama using the bundled
sample file. Note that first execuion can take few seconds if Ollama did not
start yet.

```bash
python -m decigrep -V "payment failed" sample.log
```

Expected output (semantic matching, not literal text search):

```text
Card payment declined for order #12345.
Payment gateway timeout caused order failures.
```

The `-V` flag also prints per-line probabilities to stderr.

> If you want to call Ollama on this VM from a different machine, run the
> server with a reachable host, e.g.
> `OLLAMA_HOST=0.0.0.0:11434 ollama serve`, and point DeciGrep at it with
> `-u http://<VM_EXTERNAL_IP>:11434`.

## 6. Clean up resources

To avoid ongoing charges after testing, stop or delete the instance.

Stop the VM (keeps disk/data, stops GPU/compute charges):

```bash
gcloud compute instances stop ollama-test-vm --zone=us-central1-a
```

Delete the VM (removes the VM and attached boot disk):

```bash
gcloud compute instances delete ollama-test-vm --zone=us-central1-a --quiet
