#!/usr/bin/env python3
"""Round-2 ERS from aiperf profile_export.jsonl.
ERS = mean(w*s_ttft + (1-w)*s_tpot); s = clamp((C-x)/(C-F),0,1)**gamma
TTFT F=200 C=6000 | TPOT F=8 C=100 | gamma=2 | w=0.5
Usage: scorer.py <artifact_dir_or_jsonl> [phase=profiling]
"""
import json, sys, os, statistics as st

F_TTFT, C_TTFT = 200.0, 6000.0
F_TPOT, C_TPOT = 8.0, 100.0
GAMMA, W = 2.0, 0.5

def s(x, F, C):
    if x is None: return 0.0
    v = (C - x) / (C - F)
    v = 0.0 if v < 0 else (1.0 if v > 1 else v)
    return v ** GAMMA

def mv(r, name):
    m = r.get('metrics', {}).get(name)
    if isinstance(m, dict): return m.get('value')
    return m

def q(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]

def main(path, phase='profiling'):
    if os.path.isdir(path): path = os.path.join(path, 'profile_export.jsonl')
    recs = []
    for line in open(path):
        line = line.strip()
        if line:
            try: recs.append(json.loads(line))
            except Exception: pass
    phases = {}
    for r in recs:
        p = r.get('metadata', {}).get('benchmark_phase')
        phases[p] = phases.get(p, 0) + 1
    print('phases:', phases)
    recs = [r for r in recs if r.get('metadata', {}).get('benchmark_phase') == phase]
    print(f'scoring {len(recs)} records (phase={phase})')
    scored, ttfts, tpots, zeros = [], [], [], 0
    isl, osl, pf = [], [], []
    for r in recs:
        md = r.get('metadata', {})
        ttft = mv(r, 'time_to_first_token')
        n = mv(r, 'output_sequence_length') or mv(r, 'output_token_count')
        lat = mv(r, 'request_latency')
        itl = mv(r, 'inter_token_latency')
        if md.get('was_cancelled') or ttft is None or not n or n <= 0:
            scored.append(0.0); zeros += 1; continue
        tpot = itl if itl is not None else ((lat - ttft) / (n - 1) if (lat is not None and n > 1) else None)
        if tpot is None: tpot = 0.0  # single-token response: no decode component
        ttfts.append(ttft); tpots.append(tpot)
        if mv(r, 'input_sequence_length'): isl.append(mv(r, 'input_sequence_length'))
        osl.append(n)
        if mv(r, 'prefill_throughput_per_user'): pf.append(mv(r, 'prefill_throughput_per_user'))
        scored.append(W * s(ttft, F_TTFT, C_TTFT) + (1 - W) * s(tpot, F_TPOT, C_TPOT))
    if not scored:
        print('nothing'); return
    ers = sum(scored) / len(scored)
    print(f"\n===== ERS = {ers:.4f}  =>  Score = {100 * ers:.2f}  (f(delta)=1)")
    print(f"n={len(scored)} zero={zeros}")
    st_t = [s(x, F_TTFT, C_TTFT) for x in ttfts]
    st_p = [s(x, F_TPOT, C_TPOT) for x in tpots]
    print(f"TTFT ms mean={st.mean(ttfts):8.0f} p50={q(ttfts,.5):8.0f} p90={q(ttfts,.9):8.0f} p95={q(ttfts,.95):8.0f} max={max(ttfts):8.0f}")
    print(f"   frac TTFT>6000ms (zero) = {sum(1 for x in ttfts if x >= 6000) / len(ttfts) * 100:5.1f}%")
    print(f"   frac TTFT<=200ms (full) = {sum(1 for x in ttfts if x <= 200) / len(ttfts) * 100:5.1f}%")
    print(f"   mean s_ttft = {st.mean(st_t):.4f}   -> {50 * st.mean(st_t):.2f} pts")
    print(f"TPOT ms mean={st.mean(tpots):8.2f} p50={q(tpots,.5):8.2f} p90={q(tpots,.9):8.2f} max={max(tpots):8.2f}")
    print(f"   frac TPOT>=100ms = {sum(1 for x in tpots if x >= 100) / len(tpots) * 100:5.1f}%")
    print(f"   mean s_tpot = {st.mean(st_p):.4f}   -> {50 * st.mean(st_p):.2f} pts")
    if isl: print(f"ISL mean={st.mean(isl):,.0f} p90={q(isl,.9):,.0f}")
    if osl: print(f"OSL mean={st.mean(osl):,.0f} p90={q(osl,.9):,.0f}")
    if pf:  print(f"prefill tok/s/user: mean={st.mean(pf):,.0f} p50={q(pf,.5):,.0f} p90={q(pf,.9):,.0f}")

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'profiling')
