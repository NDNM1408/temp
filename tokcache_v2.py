#!/usr/bin/env python3
"""Incremental prompt tokenisation, hooked where this vLLM actually tokenises.

The round-1 patch cached token ids per chat-template segment and was measured
here at 13.9x with bit-identical output. Its only problem is where it hooks:
`TokenizersBackend.encode`, which vLLM 0.25 reaches only when the chat template
is rendered with `tokenize=False`. Probing the two routes showed

    apply_chat_template(tokenize=True)           -> __call__ , _encode_plus
    apply_chat_template(tokenize=False)+encode    -> encode  , _encode_plus

so `_encode_plus` is the one call both routes pass through, and hooking it makes
the question of which route vLLM takes irrelevant. That matters because the
failure mode of hooking the wrong place is silence: the patch installs, reports
success, and never runs.

`_encode_plus` returns a BatchEncoding rather than a list, so the wrapper builds
one. Everything unusual -- a text pair, tensors, padding, real truncation, split
words -- falls through to the original, as does any exception. Losing the speedup
is acceptable; returning different ids is not.

Correctness is the whole point, so the test checks all three public entry points
against unpatched output, token by token.

    python3 tokcache_v2.py [model_dir]
"""
from __future__ import annotations

import sys
import threading
import time

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/srv/contest-workspace/models/Qwen3.5-122B-A10B-FP8"
TURNS = 20
CHARS_PER_TURN = 16000

stats = {"seen": 0, "engaged": 0, "reused": 0, "fellback": 0}


def install(tok_cls, marker: str, min_chars: int = 2000, max_entries: int = 256):
    """Hook _encode_plus with a marker-segment prefix cache."""
    from transformers.tokenization_utils_base import BatchEncoding

    if getattr(tok_cls, "_tokcache_v2", False):
        return
    orig = tok_cls._encode_plus
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

    def ids_of(self, seg, asp):
        return orig(self, seg, None, add_special_tokens=asp)["input_ids"]

    def patched(self, text, text_pair=None, add_special_tokens=True, **kw):
        stats["seen"] += 1
        # Anything that changes the shape of the answer goes to the original.
        simple = (
            text_pair is None
            and isinstance(text, str)
            and len(text) >= min_chars
            and isinstance(add_special_tokens, bool)
            and not kw.get("is_split_into_words")
            and kw.get("return_tensors") is None
            and str(kw.get("padding_strategy", "DO_NOT_PAD")).endswith("DO_NOT_PAD")
            and str(kw.get("truncation_strategy", "DO_NOT_TRUNCATE")).endswith("DO_NOT_TRUNCATE")
        )
        if not simple:
            return orig(self, text, text_pair, add_special_tokens=add_special_tokens, **kw)
        try:
            base_str, ids = find_base(text, add_special_tokens)
            if ids is None:
                ids = []
            else:
                stats["reused"] += 1
            stats["engaged"] += 1
            prev = len(base_str)
            bounds = []
            i = text.find(marker, prev)
            while i >= 0:
                bounds.append(i + len(marker))
                i = text.find(marker, i + len(marker))
            for b in bounds:
                asp = add_special_tokens if prev == 0 else False
                ids = ids + ids_of(self, text[prev:b], asp)
                store(text[:b], ids, add_special_tokens)
                prev = b
            tail = text[prev:]
            if tail:
                asp = add_special_tokens if prev == 0 else False
                ids = ids + ids_of(self, tail, asp)
            ml = kw.get("max_length")
            if ml is not None and len(ids) > ml:
                return orig(self, text, text_pair,
                            add_special_tokens=add_special_tokens, **kw)
            out = {"input_ids": ids}
            if kw.get("return_attention_mask") is not False:
                out["attention_mask"] = [1] * len(ids)
            return BatchEncoding(out)
        except Exception as exc:
            stats["fellback"] += 1
            print(f"  [fell back: {type(exc).__name__}: {exc}]")
            return orig(self, text, text_pair, add_special_tokens=add_special_tokens, **kw)

    tok_cls._encode_plus = patched
    tok_cls._tokcache_v2 = True


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    print(f"tokenizer: {type(tok).__name__}")

    msgs, prompts = [], []
    for t in range(TURNS):
        msgs.append({"role": "user",
                     "content": f"turn {t}: " + ("lorem ipsum dolor sit amet " * (CHARS_PER_TURN // 27))})
        msgs.append({"role": "assistant",
                     "content": f"reply {t}: " + ("consectetur adipiscing elit " * 40)})
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    # Three public entry points, because vLLM's choice between them is a config
    # detail and the patch has to be correct through all of them.
    routes = {
        "encode(str)": lambda p: tok.encode(p),
        "__call__(str)": lambda p: tok(p)["input_ids"],
        "apply_chat_template(tokenize=True)": None,   # filled in below, needs msgs
    }

    def act_ids(i):
        return tok.apply_chat_template(msgs_upto[i], tokenize=True, add_generation_prompt=True)

    msgs_upto = [msgs[: 2 * (i + 1)] for i in range(TURNS)]

    print(f"last prompt: {len(prompts[-1]):,} chars / {len(tok.encode(prompts[-1])):,} tokens")

    print("\n--- unpatched")
    ref = {}
    t_ref = {}
    for name in routes:
        t0 = time.perf_counter()
        if name == "apply_chat_template(tokenize=True)":
            ref[name] = [act_ids(i) for i in range(TURNS)]
        else:
            ref[name] = [routes[name](p) for p in prompts]
        t_ref[name] = (time.perf_counter() - t0) * 1000
        print(f"  {name:<36} {t_ref[name]:8.1f} ms")

    install(type(tok), "<|im_end|>\n")
    print("\n--- patched on _encode_plus")
    got, t_got = {}, {}
    for name in routes:
        t0 = time.perf_counter()
        if name == "apply_chat_template(tokenize=True)":
            got[name] = [act_ids(i) for i in range(TURNS)]
        else:
            got[name] = [routes[name](p) for p in prompts]
        t_got[name] = (time.perf_counter() - t0) * 1000
        print(f"  {name:<36} {t_got[name]:8.1f} ms   "
              f"{t_ref[name]/max(t_got[name],1e-9):5.2f}x")
    print(f"  _encode_plus seen {stats['seen']}, engaged {stats['engaged']}, "
          f"reused {stats['reused']}, fell back {stats['fellback']}")

    print("\n" + "=" * 66)
    allgood = True
    for name in routes:
        a, b = ref[name], got[name]
        bad = [i for i, (x, y) in enumerate(zip(a, b)) if list(x) != list(y)]
        if bad:
            allgood = False
            i = bad[0]
            print(f"EXACT {name:<36} NO -- {len(bad)}/{TURNS} differ, first turn {i} "
                  f"(len {len(a[i])} vs {len(b[i])})")
        else:
            print(f"EXACT {name:<36} yes -- all {TURNS} bit-identical")
    print(f"\nVERDICT: {'usable' if allgood and stats['reused'] else 'NOT usable'}"
          f"  (reused a cached prefix {stats['reused']} times)")


if __name__ == "__main__":
    main()
