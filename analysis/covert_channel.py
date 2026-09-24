"""How much of the log2 n bits per response that Proposition 1 allows can a colluding scorer realise?

Proposition 1 gives q(y) <= n p_s(y) for ANY score, so the mutual information between any message and
one served response is at most log2 n bits, and m responses carry at most m log2 n. This measures how
close a simple keyed-hash scorer comes to that rate over the committed anchor pools, and how many
responses it needs to move a protected 50-token window (results/covert_channel_note.md). Every served
response is an unmodified anchor draw; the message rides in WHICH draw is served, never in its text.

Rule. The scorer and the receiver share a key. A draw's keyed hash u in [0,1): [0, eps) is an erasure
cell, [eps, 1) is cut into 2^b equal cells. Serve a draw whose cell equals the next b message bits;
else a draw in the erasure cell (the receiver skips it); else the lowest cell (an undetected error).
(b, eps) come from the i.i.d. formulas alone, never from the simulation.

CPU only. Usage: .venv/bin/python analysis/covert_channel.py --out results
"""
import argparse
import csv
import hashlib
import hmac
import json
import lzma
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = b"fixed-demo-key-for-reproducible-hashing"
NS = (2, 4, 8, 16, 32, 64)
HDR = 16                      # bits of length header in front of every message


def analytic(n, b, eps):
    """(P(some draw matches the wanted cell), P(no match and no erasure draw)) for n i.i.d. hashes."""
    return 1 - (1 - (1 - eps) * 2 ** -b) ** n, ((1 - eps) * (1 - 2 ** -b)) ** n


def choose(n, max_err=1e-3):
    """argmax over (b, eps) of b * P(match), subject to P(undetected error) <= max_err."""
    best = None
    for b in range(1, int(math.log2(n)) + 1):
        for e in range(100):
            pm, pe = analytic(n, b, e / 100)
            if pe <= max_err and (best is None or b * pm > best[0]):
                best = (b * pm, b, e / 100)
    return best[1], best[2]


def u(t, dig):
    return int.from_bytes(hmac.new(KEY, t.to_bytes(8, "big") + dig, hashlib.sha256).digest()[:8],
                          "big") / 2 ** 64


def cell(x, b, eps):
    return None if x < eps else min(int((x - eps) / (1 - eps) * 2 ** b), 2 ** b - 1)


def serve(t, digs, want, b, eps):
    cells = [cell(u(t, d), b, eps) for d in digs]
    for kind, test in (("match", lambda c: c == want), ("erasure", lambda c: c is None)):
        for i, c in enumerate(cells):
            if test(c):
                return i, kind
    return min(range(len(cells)), key=lambda i: cells[i]), "error"


def receive(t, dig, b, eps):
    c = cell(u(t, dig), b, eps)
    return None if c is None else format(c, f"0{b}b")


def bits_of(raw):
    return format(len(raw), f"0{HDR}b") + "".join(format(x, "08b") for x in raw)


def raw_of(bits):
    n = int(bits[:HDR], 2)
    return bytes(int(bits[HDR + 8 * i: HDR + 8 * i + 8], 2) for i in range(n))


def send(msg_bits, pools, n, b, eps, rng, t0=0):
    """Push msg_bits through the channel. Returns (responses used, bits received, undetected errors)."""
    got, t, errs = "", t0, 0
    while len(got) < len(msg_bits):
        digs = rng.sample(pools[rng.randrange(len(pools))], n)
        want = int(msg_bits[len(got):len(got) + b].ljust(b, "0"), 2)
        i, kind = serve(t, digs, want, b, eps)
        rec = receive(t, digs[i], b, eps)
        errs += kind == "error"
        t += 1
        if rec is not None:
            got += rec
    return t - t0, got[:len(msg_bits)], errs


def load_pools(gen_dir):
    from analysis.selection_decoding import load_candidates
    c = load_candidates(gen_dir, deecho=True)
    return [[hashlib.sha256(x[3].encode()).digest() for x in v] for _, v in sorted(c.items())]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", default="output/phase5/sel_anchor64")
    ap.add_argument("--tokenizer", default="jacquelinehe/tinycomma-1.8b-llama3-tokenizer")
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--odometer-nats", type=float, default=400.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    pools = load_pools(a.gen_dir)
    distinct = sum(len(set(p)) for p in pools) / sum(len(p) for p in pools)
    print(f"[cc] {len(pools)} pools, distinct-draw fraction {distinct:.4f}", flush=True)

    from transformers import AutoTokenizer
    from analysis.selection_extraction import build
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    passages = build(tok, "data", "attack_train", 100, 20)
    wins = [tok.decode(tok(p["target"], add_special_tokens=False).input_ids[:a.window],
                       skip_special_tokens=True) for p in passages]
    print(f"[cc] {len(wins)} protected {a.window}-token windows", flush=True)

    rows, prows = [], []
    for n in NS:
        b, eps = choose(n)
        pm_a, pe_a = analytic(n, b, eps)
        tally = {"match": 0, "erasure": 0, "error": 0}
        for t in range(a.reps):
            _, kind = serve(t, rng.sample(pools[rng.randrange(len(pools))], n), rng.getrandbits(b), b, eps)
            tally[kind] += 1
        rate = b * tally["match"] / a.reps
        # every window, end to end, under two public codes: raw UTF-8 and lzma (no model)
        per_enc = {}
        for enc in ("utf8", "lzma"):
            used, ok, nbits, errs = [], 0, [], 0
            for w in wins:
                raw = w.encode() if enc == "utf8" else lzma.compress(
                    w.encode(), format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9}])
                mb = bits_of(raw)
                q, got, e = send(mb, pools, n, b, eps, rng, t0=10 ** 9 + len(prows) * 10 ** 6)
                dec = raw_of(got)
                if enc == "lzma":
                    try:
                        dec = lzma.decompress(dec, format=lzma.FORMAT_RAW,
                                              filters=[{"id": lzma.FILTER_LZMA2, "preset": 9}])
                    except lzma.LZMAError:
                        dec = b""
                ok += dec == w.encode()
                used.append(q)
                nbits.append(len(mb))
                errs += e
            per_enc[enc] = (sum(used) / len(used), sum(nbits) / len(nbits), ok, errs)
            prows.append(dict(n=n, encoding=enc, windows=len(wins), mean_bits=round(sum(nbits) / len(nbits), 1),
                              mean_responses=round(sum(used) / len(used), 1),
                              responses_at_capacity=round(sum(nbits) / len(nbits) / math.log2(n), 1),
                              decoded_exactly=ok, undetected_errors=errs))
            print(f"[cc] n={n:2d} {enc:5s}: {per_enc[enc][1]:.0f} bits, {per_enc[enc][0]:.1f} responses, "
                  f"{ok}/{len(wins)} exact", flush=True)
        rows.append(dict(n=n, b=b, eps=eps, log2_n=round(math.log2(n), 4),
                         bits_per_response=round(rate, 4), analytic_bits=round(b * pm_a, 4),
                         efficiency=round(rate / math.log2(n), 4),
                         match_rate=round(tally["match"] / a.reps, 4),
                         erasure_rate=round(tally["erasure"] / a.reps, 4),
                         undetected_error_rate=round(tally["error"] / a.reps, 6),
                         analytic_error_bound=round(pe_a, 6),
                         odometer_max_responses=int(a.odometer_nats // math.log(n)),
                         odometer_max_bits=round(a.odometer_nats / math.log(2), 1),
                         distinct_draw_frac=round(distinct, 4)))
    os.makedirs(a.out, exist_ok=True)
    for name, data in (("covert_channel.csv", rows), ("covert_channel_windows.csv", prows)):
        with open(os.path.join(a.out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(data)
    for r in rows:
        print(r)
    print(f"wrote {a.out}/covert_channel.csv, covert_channel_windows.csv")


if __name__ == "__main__":
    main()
