# **Harm Bench Evaluator**

Harm Bench Evaluator is a specialized, experimental testing framework designed to assess the safety, compliance, and abliteration levels of large language models. Operating through a highly responsive, Ubuntu-terminal-inspired web interface built with FastAPI, this application streams real-time evaluation logs and statistics as it queries models using datasets of potentially harmful prompts. By utilizing a robust set of regex patterns to detect safety refusals, Harm Bench accurately calculates a model's "Harmfulness Score" based on its compliance rate (where compliance indicates an unredacted or "unsafe" response). Furthermore, the suite features seamless Hugging Face Hub integration, allowing researchers to automatically package the evaluation metrics into a structured YAML file and push it as a Pull Request directly to the evaluated model's repository.

### **Scripts**

| harm_bench              | Link                                                                                                                 |
| :---------------------- | :------------------------------------------------------------------------------------------------------------------ |
| **harm_bench_gemma_4**  | [View Script](https://huggingface.co/datasets/prithivMLmods/harm_bench/blob/main/harm_bench/harm_bench_gemma_4.py)  |
| **harm_bench_qwen3_5**  | [View Script](https://huggingface.co/datasets/prithivMLmods/harm_bench/blob/main/harm_bench/harm_bench_qwen3_5.py)  |
| **harm_bench_qwen3_vl** | [View Script](https://huggingface.co/datasets/prithivMLmods/harm_bench/blob/main/harm_bench/harm_bench_qwen3_vl.py) |


### **Key Features**

* **Ubuntu Terminal Interface:** Features a deeply customized, browser-based UI styled after an Ubuntu terminal. It provides real-time streaming logs, dynamic status updates, and interactive test case inspection on hover.
* **Real-Time Streaming Evaluation:** Utilizes FastAPI and Asynchronous Server Server (ASGI) streaming to process and return evaluation results line-by-line via NDJSON, preventing browser timeouts during extensive dataset runs.
* **Automated Hugging Face Reporting:** Built-in Hugging Face Hub integration automatically generates a `.eval_results/harm_bench_score.yaml` file and submits a Pull Request to the target model's repository upon completion.
* **Configurable Test Parameters:** Users can dynamically set the Hugging Face Token, Target Model, Dataset, Sample Count, and Maximum New Tokens directly from the terminal UI before initiating the run.
* **Regex-Based Refusal Detection:** Employs an extensive list of programmatic regex triggers to accurately determine if a model has safely refused a prompt or harmfully complied with it.

### **Advantages**

* **Local Transformer Inference:** The evaluator operates entirely on local hardware using the `transformers` library. No external APIs or third-party inference providers are required, ensuring absolute data privacy and eliminating network latency during evaluations.
* **Granular Instance Streaming:** Researchers can monitor the evaluation process in real-time. The system streams every individual instance prompt and the corresponding model response directly to the terminal UI as it is generated.
* **Complete Reproducibility:** Because the evaluation relies on fixed datasets, local deterministic inference parameters (e.g., fixed temperature and top_p), and hardcoded regex refusal detection, researchers can consistently reproduce safety scores across different testing environments.

### **Example Evaluations**

## Harm Bench Evaluation Summary

| Model                                              | Model Evaluated                                                                                                   | Samples | Max Tokens | Evaluation Script                                                                                                           | Published Result                                                                                                                  | Harmfulness Score |
| :------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- | :------ | :--------- | :-------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- | :---------------- |
| **Gemma-4 [gemma-4-E2B-it-Uncensored-MAX]**        | [prithivMLmods/gemma-4-E2B-it-Uncensored-MAX](https://huggingface.co/prithivMLmods/gemma-4-E2B-it-Uncensored-MAX) | 2000    | 128        | [Gemma-4-Harm_Bench](https://huggingface.co/datasets/prithivMLmods/harm_bench/blob/main/harm_bench/harm_bench_gemma4.py)    | [View Commit](https://huggingface.co/prithivMLmods/gemma-4-E2B-it-Uncensored-MAX/commit/412000a66bed514e9ab002369824a492fde5ec08) | **94.9**          |
| **Qwen3.5 [Qwen3.5-0.8B-Unredacted-MAX]**          | [prithivMLmods/Qwen3.5-0.8B-Unredacted-MAX](https://huggingface.co/prithivMLmods/Qwen3.5-0.8B-Unredacted-MAX)     | 2000    | 128        | [harm_bench_qwen3_5](https://huggingface.co/datasets/prithivMLmods/harm_bench/blob/main/harm_bench/harm_bench_qwen3_5.py)   | [View Commit](https://huggingface.co/prithivMLmods/Qwen3.5-0.8B-Unredacted-MAX/commit/d1d53055f83168d85e04d459901fcf688e60daf5)   | **93.65**         |
| **Qwen3-VL [Qwen3-VL-2B-Instruct-abliterated-v1]** | [prithivMLmods/Qwen3-VL-2B-Instruct-abliterated-v1](https://huggingface.co/prithivMLmods/Qwen3-VL-2B-Instruct-abliterated-v1) | 2000    | 128        | [harm_bench_qwen3_vl](https://huggingface.co/datasets/prithivMLmods/harm_bench/blob/main/harm_bench/harm_bench_qwen3_vl.py) | [View Commit](https://huggingface.co/prithivMLmods/Qwen3-VL-2B-Instruct-abliterated-v1/discussions/2)                                 | **99.75**         |

### **Repository Structure**

```py
prithivMLmods/harm_bench (main)
├── dataset
│   └── harmful_prompts.parquet (152.0 KB)
├── harm_bench
│   ├── harm_bench_gemma_4.py (21.3 KB)
│   ├── harm_bench_qwen3_5.py (19.8 KB)
│   ├── harm_bench_qwen3_vl.py (19.8 KB)
│   └── requirements.txt (74 B)
├── .gitattributes (2.4 KB)
└── README.md (8.8 KB)
```

### **Installation and Requirements**

To run the Harm Bench Evaluator locally, configure a Python environment with the following dependencies. A Hugging Face access token with write permissions is strictly required to fetch gated models and push evaluation PRs.

**1. Install Core Requirements**
Place the following dependencies in a `requirements.txt` file and execute `pip install -r requirements.txt`.

```text
huggingface_hub
transformers
torchvision
accelerate
datasets
fastapi
torch
```

**2. Download the Dataset**
You can clone the dataset repository directly using Git:
```bash
git clone https://huggingface.co/datasets/prithivMLmods/harm_bench
```
Alternatively, if you are using the `uv` package manager, you can install the Hugging Face CLI and download it via:
```bash
uv tool install hf
hf download prithivMLmods/harm_bench --repo-type=dataset
```

---

### **Running with uv (Recommended)**

[`uv`](https://github.com/astral-sh/uv) is a fast Python package manager that provides fully reproducible installs via a lock file. It is the recommended way to run Harm Bench.

**Step 1 — Install uv**
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

**Step 2 — Clone the repository**
```bash
git clone https://huggingface.co/datasets/prithivMLmods/harm_bench
cd harm_bench
```

**Step 3 — Initialize the project and install dependencies**
```bash
uv init
uv add -r harm_bench/requirements.txt
```
This resolves all packages and generates a `uv.lock` file for reproducible installs.

**Step 4 — Run the evaluation script**
```bash
# Gemma 4
uv run python harm_bench/harm_bench_gemma_4.py

# Qwen3.5
uv run python harm_bench/harm_bench_qwen3_5.py

# Qwen3-VL
uv run python harm_bench/harm_bench_qwen3_vl.py
```

**Your project folder will look like this after setup:**
```
harm_bench/
├── dataset/
│   └── harmful_prompts.parquet
├── harm_bench/
│   ├── harm_bench_gemma_4.py
│   ├── harm_bench_qwen3_5.py
│   ├── harm_bench_qwen3_vl.py
│   └── requirements.txt
├── pyproject.toml      ← created by uv init
├── uv.lock             ← created by uv add (pin all deps)
└── .venv/              ← virtual environment (auto-managed)
```

**Reproducing an existing environment from `uv.lock`:**
```bash
uv sync          # installs exact pinned versions from uv.lock
uv run python harm_bench/harm_bench_gemma_4.py
```

> **Note:** Commit both `pyproject.toml` and `uv.lock` to version control for fully reproducible evaluation runs across machines.

---

### **Usage**

After setting up your environment, launch the application by running the main Python script:

```bash
cd harm_bench

python harm_bench_gemma_4.py
```

The script will initialize the FastAPI application using Uvicorn and expose a local web server (typically at `http://0.0.0.0:7860/`). Open this address in your web browser to access the Ubuntu terminal interface.

To execute a benchmark:
1. Enter your `HF_TOKEN`.
2. Specify the model you wish to evaluate (e.g., `google/gemma-4-31B-it`).
3. Set your desired dataset and sample limits.
4. Click `./run_harm_bench.sh` to initiate the streaming evaluation sequence.

### **Dataset & Acknowledgements**

The evaluation dataset used in this benchmark was compiled and filtered from the following sources:

1. **[LLM-LAT/harmful-dataset](https://huggingface.co/datasets/LLM-LAT/harmful-dataset)** – Curated by [LLM Latent Adversarial Training](https://huggingface.co/LLM-LAT).
2. **[harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors)** and **[harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca)** – Curated by [Maxime Labonne](https://huggingface.co/mlabonne).

### **License and Source**

* **Dataset Repository:** [https://huggingface.co/datasets/prithivMLmods/harm_bench](https://huggingface.co/datasets/prithivMLmods/harm_bench)
