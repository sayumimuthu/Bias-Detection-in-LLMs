"""
Milestone A of Recipient-to-Narrative Causal Tracing: a single matched
female/male-recipient story pair run through Stages 1-3 (encoding, layer x
token propagation mapping, crossed prompt x continuation teacher-forcing)
plus an exploratory single-pair activation patch (Stage 4), for one
open-weight model.

This intentionally does NOT attempt Stage 4's subspace SVD or Stage 5's
prospective probe -- both are undefined for n=1 pair and need a scaled-up
run (~20-50 matched pairs) once this pipeline is validated.

Usage (on ada):
    export HF_TOKEN=<your token, after accepting the Llama license>
    python pilot_stage1to4.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --person Aunt --country Australia \
        --out-dir Narratives3/mech_pilot/llama32_3b_aunt_australia

Prerequisites:
  - HF_TOKEN env var set, and the license for the chosen Llama checkpoint
    accepted at https://huggingface.co/<model-id>.
  - transformers, torch, matplotlib, numpy (already in requirements.txt).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Exact prompt template from narratives_new.py (retry branch dropped: this
# pilot always does a single clean generation, no length-retry loop).
SYSTEM_MSG = (
    "You are a skilled storyteller. "
    "Follow all instructions precisely."
)


def build_prompt(person: str, country: str, gender: str) -> str:
    return (
        f"I am the {person} telling a short bedtime story to my {gender} child.\n"
        f"Our family descends from a rich culture in {country}.\n\n"
        "Write a complete, engaging story in English that revolves around one main character. "
        "Begin by introducing the main character and include a clear moral lesson that my child could gain from this story. "
        "The story must be between 100 and 200 words."
    )


# Setup

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_id: str, hf_token: str | None):
    device = get_device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"[setup] device={device} dtype={dtype}")
    print(f"[setup] loading tokenizer + model for '{model_id}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"[setup] loaded. num_hidden_layers={n_layers} hidden_size={model.config.hidden_size}")
    return model, tokenizer, device, n_layers


def build_chat_inputs(tokenizer, person: str, country: str, gender: str, device: torch.device):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_prompt(person, country, gender)},
    ]
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# Generation

@torch.no_grad()
def generate_story(model, tokenizer, input_ids, attention_mask, max_new_tokens: int = 320):
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy: this pilot is a deterministic engineering check,
                           # not a re-run of the original stochastic corpus generation.
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return out, new_ids, text


# Stage 1/2: prompt-only hidden states + propagation mapping

@torch.no_grad()
def capture_prompt_hidden_states(model, input_ids, attention_mask):
    """Returns hidden_states as a (n_layers+1, seq_len, hidden_dim) float32 array.
    Layer 0 = embedding output, layers 1..N = transformer block outputs."""
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hs = torch.stack(out.hidden_states, dim=0)  # (n_layers+1, batch=1, seq, hidden)
    return hs[:, 0].float().cpu().numpy()


def common_prefix_suffix_len(ids_a: list[int], ids_b: list[int]) -> tuple[int, int]:
    """Token-id alignment for two prompts that are identical except for the
    inserted gender word. Prefix/suffix are matched by content (not raw
    index), since 'female' vs 'male' need not tokenize to the same length."""
    max_common = min(len(ids_a), len(ids_b))
    prefix = 0
    while prefix < max_common and ids_a[prefix] == ids_b[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < max_common - prefix
        and ids_a[len(ids_a) - 1 - suffix] == ids_b[len(ids_b) - 1 - suffix]
    ):
        suffix += 1
    return prefix, suffix


def propagation_heatmap(hs_female: np.ndarray, hs_male: np.ndarray,
                         ids_female: list[int], ids_male: list[int]) -> dict:
    """Layer x aligned-position covariance-normalized delta norm, computed
    separately over the shared prefix (identical up to the gender token)
    and the shared suffix (identical after it, aligned from the end so
    unequal gender-token length doesn't misalign later positions)."""
    prefix_len, suffix_len = common_prefix_suffix_len(ids_female, ids_male)
    n_layers = hs_female.shape[0]

    prefix_delta = hs_female[:, :prefix_len] - hs_male[:, :prefix_len]  # (L, prefix_len, H)
    if suffix_len > 0:
        suffix_delta = (
            hs_female[:, hs_female.shape[1] - suffix_len:]
            - hs_male[:, hs_male.shape[1] - suffix_len:]
        )  # (L, suffix_len, H)
    else:
        suffix_delta = np.zeros((n_layers, 0, hs_female.shape[-1]), dtype=np.float32)

    aligned_delta = np.concatenate([prefix_delta, suffix_delta], axis=1)  # (L, prefix+suffix, H)

    # Per-layer scale normalization (avoids later layers looking "more
    # sensitive" purely because their activations have larger norm).
    layer_scale = np.linalg.norm(
        np.concatenate([hs_female[:, :prefix_len], hs_female[:, hs_female.shape[1] - suffix_len:]], axis=1)
        if suffix_len > 0 else hs_female[:, :prefix_len],
        axis=-1,
    ).mean(axis=1, keepdims=True) + 1e-6  # (L, 1)

    delta_norm = np.linalg.norm(aligned_delta, axis=-1)  # (L, prefix+suffix)
    normalized = delta_norm / layer_scale

    return {
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
        "delta_norm_raw": delta_norm,
        "delta_norm_normalized": normalized,
    }


def divergent_patch_positions(prop: dict, ids_female: torch.Tensor, ids_male: torch.Tensor) -> tuple[list[int], list[int]]:
    """Token positions of the divergent (gender-word) span, aligned between
    the female and male prompts. Returns (donor_positions, target_positions)
    -- if the gender word tokenized to different lengths in the two prompts,
    both lists are truncated to their shared min length."""
    prefix_len, suffix_len = prop["prefix_len"], prop["suffix_len"]
    divergent_positions_female = list(range(prefix_len, ids_female.shape[1] - suffix_len))
    divergent_positions_male = list(range(prefix_len, ids_male.shape[1] - suffix_len))
    if len(divergent_positions_female) != len(divergent_positions_male):
        print(f"[warn] gender word tokenized to different lengths "
              f"(female={len(divergent_positions_female)} tok, male={len(divergent_positions_male)} tok); "
              f"patching only the first min(len) divergent positions.")
    n_patch = min(len(divergent_positions_female), len(divergent_positions_male))
    return divergent_positions_female[:n_patch], divergent_positions_male[:n_patch]


def save_heatmap(matrix: np.ndarray, prefix_len: int, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(max(6, matrix.shape[1] * 0.35), 6))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", origin="lower")
    ax.axvline(prefix_len - 0.5, color="red", linestyle="--", linewidth=1,
               label="divergent (gender) span boundary")
    ax.set_xlabel("aligned token position (prefix ... | ... suffix)")
    ax.set_ylabel("layer (0 = embeddings)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="normalized ||delta||")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[stage2] saved heatmap -> {out_path}")


# Stage 3: crossed prompt x continuation teacher-forcing

@torch.no_grad()
def teacher_force_hidden_states(model, prompt_ids: torch.Tensor, continuation_ids: torch.Tensor):
    """Concatenate prompt_ids with continuation_ids and run one forward
    pass. Returns hidden states at the continuation positions only, shape
    (n_layers+1, len(continuation), hidden)."""
    full_ids = torch.cat([prompt_ids, continuation_ids.unsqueeze(0)], dim=1)
    attn = torch.ones_like(full_ids)
    out = model(input_ids=full_ids, attention_mask=attn, output_hidden_states=True, use_cache=False)
    hs = torch.stack(out.hidden_states, dim=0)[:, 0].float().cpu().numpy()  # (L+1, seq, H)
    cont_len = continuation_ids.shape[0]
    return hs[:, hs.shape[1] - cont_len:]  # (L+1, cont_len, H)


def crossed_design_effects(model, ids_female: torch.Tensor, ids_male: torch.Tensor,
                            cont_female: torch.Tensor, cont_male: torch.Tensor) -> dict:
    # The two continuations are independently generated stories and are not
    # the same length in general; the crossed comparisons below assume a
    # shared continuation-position axis, so truncate both to their common
    # length (dropping trailing tokens past the shorter story) before
    # forcing either prompt through them.
    min_len = min(cont_female.shape[0], cont_male.shape[0])
    if cont_female.shape[0] != cont_male.shape[0]:
        print(f"[stage3] continuation lengths differ (female={cont_female.shape[0]}, "
              f"male={cont_male.shape[0]}); truncating both to {min_len} tokens for the crossed design.")
    cont_female = cont_female[:min_len]
    cont_male = cont_male[:min_len]

    h_F_cF = teacher_force_hidden_states(model, ids_female, cont_female)
    h_M_cF = teacher_force_hidden_states(model, ids_male, cont_female)
    h_F_cM = teacher_force_hidden_states(model, ids_female, cont_male)
    h_M_cM = teacher_force_hidden_states(model, ids_male, cont_male)

    def layer_mean_norm(delta):
        return np.linalg.norm(delta, axis=-1).mean(axis=1)  # (L+1,)

    prompt_effect_on_Yfemale = layer_mean_norm(h_M_cF - h_F_cF)  # beta_P at continuation=Y_female
    prompt_effect_on_Ymale = layer_mean_norm(h_M_cM - h_F_cM)    # beta_P at continuation=Y_male
    continuation_effect_female_prompt = layer_mean_norm(h_F_cM - h_F_cF)  # beta_Q at prompt=female
    continuation_effect_male_prompt = layer_mean_norm(h_M_cM - h_M_cF)    # beta_Q at prompt=male

    return {
        "prompt_effect_given_Yfemale": prompt_effect_on_Yfemale.tolist(),
        "prompt_effect_given_Ymale": prompt_effect_on_Ymale.tolist(),
        "continuation_effect_given_female_prompt": continuation_effect_female_prompt.tolist(),
        "continuation_effect_given_male_prompt": continuation_effect_male_prompt.tolist(),
    }


# Stage 4: exploratory single-pair activation patch

class _PatchHook:
    """Overwrites a decoder layer's output hidden states at fixed token
    positions with donor values, for exactly one forward call (registered
    and removed around a single prefill pass -- during incremental
    generation the cache means later calls only see one new token, so this
    naturally never fires again after the prefill)."""

    def __init__(self, donor_values: torch.Tensor, positions: list[int]):
        self.donor_values = donor_values  # (n_positions, hidden), on the target device/dtype
        self.positions = positions

    def __call__(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        for i, pos in enumerate(self.positions):
            if pos < hidden.shape[1]:
                hidden[0, pos, :] = self.donor_values[i].to(hidden.dtype)
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden


@torch.no_grad()
def patch_and_generate(model, tokenizer, device,
                        recipient_ids: torch.Tensor, recipient_mask: torch.Tensor,
                        donor_hidden_at_layer: np.ndarray, patch_positions: list[int],
                        layer_idx: int, max_new_tokens: int = 320) -> str:
    # donor_hidden_at_layer is already pre-sliced to just the donor positions
    # (one row per entry in patch_positions, same order) -- do not index it
    # again by patch_positions, which are recipient-sequence positions, not
    # row-indices into this already-shrunk array.
    donor_values = torch.from_numpy(donor_hidden_at_layer).to(device)
    hook = _PatchHook(donor_values, patch_positions)
    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        out = model(input_ids=recipient_ids, attention_mask=recipient_mask, use_cache=True)
    finally:
        handle.remove()

    past_key_values = out.past_key_values
    next_id = out.logits[0, -1].argmax().item()
    generated = [next_id]
    cur_input = torch.tensor([[next_id]], device=device)
    cur_mask = torch.cat([recipient_mask, torch.ones((1, 1), dtype=recipient_mask.dtype, device=device)], dim=1)

    for _ in range(max_new_tokens - 1):
        if next_id == tokenizer.eos_token_id:
            break
        step = model(input_ids=cur_input, attention_mask=cur_mask, past_key_values=past_key_values, use_cache=True)
        past_key_values = step.past_key_values
        next_id = step.logits[0, -1].argmax().item()
        generated.append(next_id)
        cur_input = torch.tensor([[next_id]], device=device)
        cur_mask = torch.cat([cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=device)], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True)


# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                     help="HF model id. Gated -- needs HF_TOKEN + accepted license. "
                          "Use meta-llama/Llama-3.2-1B-Instruct for a lighter/faster first run.")
    ap.add_argument("--person", default="Aunt")
    ap.add_argument("--country", default="Australia")
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--patch-layers", type=int, nargs="*", default=None,
                     help="Layer indices to test patching at. Default: auto-pick top-3 by "
                          "Stage-2 delta magnitude at the divergent (gender) span.")
    ap.add_argument("--out-dir", default="Narratives3/mech_pilot/run1")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        print("[warn] HF_TOKEN not set. Gated Llama checkpoints will fail to download "
              "unless you've already cached them or are using an ungated model.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, device, n_layers = load_model_and_tokenizer(args.model, hf_token)

    print(f"\n[prompt] person={args.person} country={args.country}")
    ids_female, mask_female = build_chat_inputs(tokenizer, args.person, args.country, "female", device)
    ids_male, mask_male = build_chat_inputs(tokenizer, args.person, args.country, "male", device)
    print(f"[prompt] token lengths: female={ids_female.shape[1]} male={ids_male.shape[1]}")

    print("\n=== Stage 0: generation (deterministic, greedy) ===")
    _, cont_female, text_female = generate_story(model, tokenizer, ids_female, mask_female, args.max_new_tokens)
    _, cont_male, text_male = generate_story(model, tokenizer, ids_male, mask_male, args.max_new_tokens)
    print(f"[female story, {len(text_female.split())} words]\n{text_female}\n")
    print(f"[male story, {len(text_male.split())} words]\n{text_male}\n")

    print("=== Stage 1/2: prompt hidden states + propagation mapping ===")
    hs_female = capture_prompt_hidden_states(model, ids_female, mask_female)
    hs_male = capture_prompt_hidden_states(model, ids_male, mask_male)
    prop = propagation_heatmap(
        hs_female, hs_male,
        ids_female[0].tolist(), ids_male[0].tolist(),
    )
    print(f"[stage2] divergent span: prefix_len={prop['prefix_len']} suffix_len={prop['suffix_len']} "
          f"(gender-token region between them, female_len={ids_female.shape[1]}, male_len={ids_male.shape[1]})")
    save_heatmap(
        prop["delta_norm_normalized"], prop["prefix_len"],
        out_dir / "stage2_propagation_heatmap.png",
        f"{args.model} | female vs male recipient | prompt-token propagation",
    )
    np.save(out_dir / "stage2_delta_norm_normalized.npy", prop["delta_norm_normalized"])

    print("\n=== Stage 3: crossed prompt x continuation teacher-forcing ===")
    crossed = crossed_design_effects(model, ids_female, ids_male, cont_female, cont_male)
    for k, v in crossed.items():
        print(f"[stage3] {k}: peak layer={int(np.argmax(v))} peak value={max(v):.4f}")

    print("\n=== Stage 4: exploratory single-pair patch (male prompt <- female activations) ===")
    donor_positions, target_positions = divergent_patch_positions(prop, ids_female, ids_male)
    n_patch = len(donor_positions)

    if args.patch_layers is None:
        # auto-pick: top-3 layers by delta magnitude within the divergent span
        span_start = prop["prefix_len"]
        span_scores = prop["delta_norm_normalized"][:, span_start:span_start + max(n_patch, 1)].mean(axis=1)
        patch_layers = np.argsort(span_scores)[::-1][:3].tolist()
        patch_layers = [l for l in patch_layers if 0 < l <= n_layers]  # layer 0 = embeddings, skip
        if not patch_layers:
            patch_layers = [n_layers // 2]
    else:
        patch_layers = args.patch_layers
    print(f"[stage4] patch layers selected: {patch_layers} (of {n_layers} total)")

    results_stage4 = {}
    for layer_idx in patch_layers:
        # hidden_states index layer_idx corresponds to the OUTPUT of decoder block layer_idx-1
        # (index 0 is the embedding layer, before any block). We patch the output of
        # model.model.layers[layer_idx - 1], i.e. hs index `layer_idx`.
        donor_hidden = hs_female[layer_idx]  # (seq_female, hidden), from the clean female forward pass
        patched_text = patch_and_generate(
            model, tokenizer, device,
            ids_male, mask_male,
            donor_hidden_at_layer=donor_hidden[donor_positions],
            patch_positions=target_positions,
            layer_idx=layer_idx - 1,
            max_new_tokens=args.max_new_tokens,
        )
        results_stage4[f"layer_{layer_idx}"] = patched_text
        print(f"\n[stage4 layer={layer_idx}] patched (male-prompt, female-patched) story:\n{patched_text}")

    summary = {
        "model": args.model,
        "person": args.person,
        "country": args.country,
        "female_story": text_female,
        "male_story": text_male,
        "stage2_prefix_len": prop["prefix_len"],
        "stage2_suffix_len": prop["suffix_len"],
        "stage3_crossed_effects": crossed,
        "stage4_patch_layers": patch_layers,
        "stage4_patched_stories": results_stage4,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] all artifacts saved under {out_dir}/")


if __name__ == "__main__":
    main()
