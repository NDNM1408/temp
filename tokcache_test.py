#!/usr/bin/env python3
"""Does the round-1 incremental tokenizer cache still work here, and is it exact?

A multi-turn agent prompt is the previous prompt plus new content, so the same
70k-token prefix is re-tokenised from scratch on every turn. The engine's prefix
cache stops it from re-*computing* attention over that prefix -- 89.9% hit -- but
nothing stops the frontend from re-tokenising it, and tokenising is superlinear:
104 ms at 65k tokens, 246 ms at 105k.

Round 1 already solved this (`submissions/P20-*/patches/sitecustomize.py`): cache
the token ids of each cumulative prefix, split new text at the chat template's
turn marker, and only tokenise what is new. The split point is the interesting
part -- it lands right after a special token, and special tokens are matched
before BPE, so no merge can straddle the boundary. That is what makes the result
bit-identical rather than merely close.

Three things decide whether it is usable, and none of them need a GPU:

  fires      the patch hooks TokenizersBackend.encode. If vLLM tokenises through
             some other call, the patch installs and then never runs -- silently.
  exact      ids must equal stock ids element for element. One differing id
             changes the model's input, and nothing anywhere would report it.
  worth it   how much time it actually saves at the real prompt size.

    python3 tokcache_test.py [model_dir]
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/srv/contest-workspace/models/Qwen3.5-122B-A10B-FP8"
TURNS = 20
CHARS_PER_TURN = 16000          # ~4k tokens, so 20 turns land near the measured median


# --- the round-1 cache, transplanted verbatim in behaviour ------------------

hits = {"calls": 0, "engaged": 0, "base_reuse": 0}


def install_tokcache(marker: str, min_chars: int = 2000, max_entries: int = 256):
    from transformers.tokenization_utils_tokenizers import TokenizersBackend as tb
    if getattr(tb, "_tokcache_installed", False):
        return tb
    orig = tb.encode
    lock = threading.Lock()
    entries: list[list] = []

    def find_base(text, asp):
        with lock:
            best_str, best_ids = "", None
            for ps, pids, pasp in entries:
                if pasp == asp and len(ps) > len(best_str) and text.startswith(ps):
                    best_str, best_ids = ps, pids
            return best_str, (list(best_ids) if best_ids is not None else None)

    def store(ps, ids, asp):
        with lock:
            for e in entries:
                if e[0] == ps and e[2] == asp:
                    return
            entries.append([ps, list(ids), asp])
            if len(entries) > max_entries:
                del entries[0]

    def patched(self, text, text_pair=None, add_special_tokens=True,
                truncation=None, max_length=None, **kw):
        hits["calls"] += 1
        if (text_pair is not None or kw or not isinstance(text, str)
                or len(text) < min_chars or not isinstance(add_special_tokens, bool)):
            return orig(self, text, text_pair, add_special_tokens=add_special_tokens,
                        truncation=truncation, max_length=max_length, **kw)
        hits["engaged"] += 1
        try:
            base_str, ids = find_base(text, add_special_tokens)
            if ids is None:
                ids = []
            else:
                hits["base_reuse"] += 1
            prev = len(base_str)
            bounds = []
            i = text.find(marker, prev)
            while i >= 0:
                bounds.append(i + len(marker))
                i = text.find(marker, i + len(marker))
            for b in bounds:
                seg = text[prev:b]
                asp = add_special_tokens if prev == 0 else False
                ids = ids + orig(self, seg, add_special_tokens=asp)
                store(text[:b], ids, add_special_tokens)
                prev = b
            tail = text[prev:]
            if tail:
                asp = add_special_tokens if prev == 0 else False
                ids = ids + orig(self, tail, add_special_tokens=asp)
            if truncation and max_length is not None and len(ids) > max_length:
                return orig(self, text, text_pair, add_special_tokens=add_special_tokens,
                            truncation=truncation, max_length=max_length, **kw)
            return ids
        except Exception as exc:
            print(f"  [patched encode fell back: {type(exc).__name__}: {exc}]")
            return orig(self, text, text_pair, add_special_tokens=add_special_tokens,
                        truncation=truncation, max_length=max_length, **kw)

    tb.encode = patched
    tb._tokcache_installed = True
    return tb


def main() -> None:
    from transformers import AutoTokenizer

    print(f"model: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f"tokenizer: {type(tok).__name__}")

    # The marker has to come from this model's template, not from round 1's.
    cfg_path = os.path.join(MODEL, "tokenizer_config.json")
    template = ""
    if os.path.exists(cfg_path):
        template = json.load(open(cfg_path)).get("chat_template") or ""
    if isinstance(template, list):                     # newer format: list of dicts
        template = " ".join(t.get("template", "") for t in template)
    for cand in ("<|im_end|>\n", "<|im_end|>", "<|eot_id|>", "</s>"):
        if cand.strip() and cand.replace("\n", "") in template:
            marker = cand
            break
    else:
        marker = "<|im_end|>\n"
    print(f"marker chosen: {marker!r}   (found in chat template: "
          f"{marker.replace(chr(10), '') in template})")

    # Growing conversation, rendered through the real template so the markers
    # in the string are the ones the server would produce.
    msgs, prompts = [], []
    for t in range(TURNS):
        msgs.append({"role": "user", "content": f"turn {t}: " + ("lorem ipsum dolor sit amet " * (CHARS_PER_TURN // 27))})
        msgs.append({"role": "assistant", "content": f"reply {t}: " + ("consectetur adipiscing elit " * 40)})
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    print(f"built {len(prompts)} prompts, last one {len(prompts[-1]):,} chars "
          f"/ {len(tok.encode(prompts[-1])):,} tokens")
    print(f"marker occurrences in last prompt: {prompts[-1].count(marker)}")

    # --- stock ---
    print("\n--- stock encode")
    stock, t_stock = [], []
    for p in prompts:
        t0 = time.perf_counter()
        stock.append(tok.encode(p))
        t_stock.append((time.perf_counter() - t0) * 1000)
    print(f"  total {sum(t_stock):8.1f} ms   last turn {t_stock[-1]:7.1f} ms")

    # --- patched ---
    install_tokcache(marker)
    print("\n--- with the incremental cache")
    patched_ids, t_patch = [], []
    for p in prompts:
        t0 = time.perf_counter()
        patched_ids.append(tok.encode(p))
        t_patch.append((time.perf_counter() - t0) * 1000)
    print(f"  total {sum(t_patch):8.1f} ms   last turn {t_patch[-1]:7.1f} ms")
    print(f"  encode calls seen by the patch: {hits['calls']}, "
          f"engaged: {hits['engaged']}, reused a cached prefix: {hits['base_reuse']}")

    # --- the three verdicts ---
    print("\n" + "=" * 60)
    if hits["calls"] == 0:
        print("FIRES    : NO -- tok.encode does not route through TokenizersBackend.encode")
    elif hits["base_reuse"] == 0:
        print("FIRES    : hooked, but never reused a prefix (marker or threshold wrong)")
    else:
        print(f"FIRES    : yes, reused a cached prefix on {hits['base_reuse']}/{len(prompts)} turns")

    bad = [i for i, (a, b) in enumerate(zip(stock, patched_ids)) if a != b]
    if bad:
        i = bad[0]
        a, b = stock[i], patched_ids[i]
        first = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"EXACT    : NO -- {len(bad)}/{len(prompts)} prompts differ; "
              f"first at turn {i}, token index {first} (len {len(a)} vs {len(b)})")
    else:
        print(f"EXACT    : yes -- all {len(prompts)} prompts bit-identical to stock")

    saved = sum(t_stock) - sum(t_patch)
    print(f"WORTH IT : total {sum(t_stock):.0f} -> {sum(t_patch):.0f} ms "
          f"({saved:+.0f} ms, {sum(t_stock)/max(sum(t_patch),1e-9):.2f}x)")
    print(f"           last turn {t_stock[-1]:.0f} -> {t_patch[-1]:.0f} ms "
          f"at {len(stock[-1]):,} tokens")

    # Which call the server actually makes matters as much as whether this works.
    before = hits["calls"]
    tok(prompts[-1])
    print(f"\ntokenizer(text) __call__ routes through the patch: "
          f"{hits['calls'] > before}")


if __name__ == "__main__":
    main()
