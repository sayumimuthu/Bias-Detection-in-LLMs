from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import gc

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")


DATA_PATH  = Path("Narratives3/clean_stories_for_analysis.csv")
OUT_DIR    = Path("Narratives3/hidden_states_by_model")

BATCH_SIZE = 2     # 2 is safe for 7B models; use 1 if OOM persists on 12B+
MAX_LENGTH = 512   # stories are ~150 words / ~200 tokens 

# model_key: HuggingFace model ID 
HF_MODEL_MAP: dict[str, str] = {
    # Open models (no token needed) 
    "ollama-qwen25-3b":    "Qwen/Qwen2.5-3B-Instruct",
    "ollama-qwen25-7b":    "Qwen/Qwen2.5-7B-Instruct",
    "ollama-mistral-7b":   "mistralai/Mistral-7B-Instruct-v0.3",
    "ollama-mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    # Gated models (need HF_TOKEN + accepted licence) 
    "ollama-llama32-1b":   "meta-llama/Llama-3.2-1B-Instruct",
    "ollama-llama32-3b":   "meta-llama/Llama-3.2-3B-Instruct",
    "ollama-llama3-70b":   "meta-llama/Meta-Llama-3-70B-Instruct",
    "ollama-llama31-8b":   "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "ollama-gemma2-2b":    "google/gemma-2-2b-it",
    "ollama-gemma3-12b":   "google/gemma-3-12b-it",
}




def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Pooling 

def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states over non-padding token positions."""
    expanded = mask.unsqueeze(-1).float()
    return (hidden * expanded).sum(1) / expanded.sum(1).clamp(min=1e-9)


# Per-model extraction 

def extract_for_model(
    model_key: str,
    hf_id: str,
    stories: list[str],
    device: torch.device,
) -> tuple[np.ndarray, int, int]:
    """
    Load `hf_id`, run all `stories` through it with output_hidden_states=True.

    Returns
    -------
    vecs       float32  (n_stories, n_layers, hidden_dim)
               layer 0  = token embedding output (before any transformer block)
               layer 1..n = transformer block outputs, 1-indexed
    n_layers   total number of layers including the embedding layer
    hidden_dim model hidden dimension
    """
    hf_token = os.environ.get("HF_TOKEN")

    print(f"  Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, token=hf_token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model weights …")
    # For 70B+ models use 4-bit quantization to fit in a single A100 80GB.
    # For smaller models bfloat16 is used directly.
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    is_large = any(tag in hf_id.lower() for tag in ["70b", "65b", "72b", "34b"])
    if is_large and device.type == "cuda":
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModel.from_pretrained(
            hf_id,
            token=hf_token,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModel.from_pretrained(
            hf_id,
            token=hf_token,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    # VLMs (e.g. Gemma3) nest text config under text_config
    cfg = model.config
    if not hasattr(cfg, "num_hidden_layers") and hasattr(cfg, "text_config"):
        cfg = cfg.text_config

    # n_layers includes the embedding layer (index 0)
    n_layers   = cfg.num_hidden_layers + 1
    hidden_dim = cfg.hidden_size
    print(f"  {n_layers} layers (0=embedding, 1..{n_layers-1}=transformer blocks)  "
          f"|  hidden_dim={hidden_dim}")

    all_vecs = np.zeros((len(stories), n_layers, hidden_dim), dtype=np.float32)

    for start in tqdm(range(0, len(stories), BATCH_SIZE),
                      desc=f"  extracting {model_key}", leave=True):
        end   = min(start + BATCH_SIZE, len(stories))
        batch = stories[start:end]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_device = next(model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # VLMs wrap text hidden states one level deeper
        hs = outputs.hidden_states
        if hs is None and hasattr(outputs, "language_model_outputs"):
            hs = outputs.language_model_outputs.hidden_states
        mask = inputs["attention_mask"]

        for layer_idx, layer_hs in enumerate(hs):
            pooled = mean_pool(layer_hs, mask)          # (B, H)
            # Replace Inf/NaN (float16 overflow) with 0 so downstream analyses don't crash
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
            all_vecs[start:end, layer_idx] = pooled.cpu().float().numpy()

    # Free GPU memory before loading next model
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return all_vecs, n_layers, hidden_dim


# Main 

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only", default=None,
        help="Process only models whose key contains this string "
             "(e.g. 'qwen', 'mistral', 'llama', 'gemma').",
    )
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}\n")

    df = pd.read_csv(DATA_PATH)
    df = df[df["word_count_compliant"] == True].reset_index(drop=True)
    print(f"{len(df):,} compliant stories  |  {df['model_key'].nunique()} models\n")

    layer_info_rows: list[dict] = []

    model_items = [
        (k, v) for k, v in HF_MODEL_MAP.items()
        if args.only is None or args.only.lower() in k.lower()
    ]

    for model_key, hf_id in model_items:
        hs_path   = OUT_DIR / f"hidden_states_{model_key}.npy"
        meta_path = OUT_DIR / f"extractor_meta_{model_key}.csv"

        if hs_path.exists():
            print(f"[skip]  {model_key} — cache exists ({hs_path.name})")
            continue

        story_mask = df["model_key"] == model_key
        if not story_mask.any():
            print(f"[skip]  {model_key} : no stories in dataset")
            continue

        sub     = df[story_mask].reset_index(drop=True)
        stories = sub["story"].tolist()
        print(f" {model_key}  ({len(stories)} stories)")
        print(f"   HF model: {hf_id}")

        try:
            vecs, n_layers, hidden_dim = extract_for_model(
                model_key, hf_id, stories, device
            )
        except Exception as e:
            err = str(e)
            print(f"  [ERROR]  {err[:120]}")
            if any(x in err for x in ("401", "403", "gated", "access", "token")):
                print(f"    Gated model. Accept the licence at:")
                print(f"    https://huggingface.co/{hf_id}")
                print(f"    Then:  export HF_TOKEN=hf_<your_token>")
            print(f"  Skipping.\n")
            continue

        # Save raw hidden states: shape fully preserved, no normalisation
        np.save(hs_path, vecs)
        sub_meta_cols = [
            "id", "model_key", "model_family", "model_params",
            "protagonist_gender", "country", "person", "word_count", "tokens",
        ]
        sub[sub_meta_cols].to_csv(meta_path, index=False)

        layer_info_rows.append({
            "model_key":   model_key,
            "hf_model_id": hf_id,
            "n_layers":    n_layers,
            "hidden_dim":  hidden_dim,
            "n_stories":   len(stories),
        })
        print(f"  Saved  {hs_path.name}  shape={vecs.shape}\n")

    # Reference table 
    info_path = OUT_DIR / "model_layer_info.csv"
    if layer_info_rows:
        info_df = pd.DataFrame(layer_info_rows)
        info_df.to_csv(info_path, index=False)
        print("\nExtracted models:")
        print(info_df.to_string(index=False))
    elif info_path.exists():
        print("\nExisting model_layer_info.csv:")
        print(pd.read_csv(info_path).to_string(index=False))

    print(f"\nAll outputs in  {OUT_DIR}/")


if __name__ == "__main__":
    main()
