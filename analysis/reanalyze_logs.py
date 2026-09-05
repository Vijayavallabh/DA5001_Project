import json, math, sys, statistics as st
from collections import defaultdict

def load(path):
    recs = {}
    for line in open(path):
        line=line.strip()
        if not line: continue
        r = json.loads(line)
        m = r["metadata"]; a = r["aggregate"]; steps = r["per_step_log"]
        Z = a["total_spend"]; B = a["final_budget"]; util = a["budget_utilization"]
        gen_len = a["generation_length_tokens"]
        max_a = 0.0; n_active=0; n_forced=0; n_free=0; L=0.0; Lmax=-1e9; Lrisky=0.0; viol_step=0; max_kt_over=0.0
        nsteps=0
        for s in steps:
            at = s["a_t"]; kt = s["k_t"]; bd = s["bd"]
            if kt is None or bd is None: continue
            nsteps+=1
            if at > max_a: max_a = at
            if at > kt + 1e-3: viol_step += 1; max_kt_over=max(max_kt_over, at-kt)
            if bd is not None:
                if bd <= 1e-6: n_forced+=1
                elif bd >= 1-1e-6: n_free+=1
                else: n_active+=1
            ps, pst = s.get("p_s_prob"), s.get("p_star_prob"); pr = s.get("p_risky_prob")
            if ps and pst and ps>0 and pst>0:
                lr = math.log(pst)-math.log(ps); L += lr; Lmax=max(Lmax, lr)
            if ps and pr and ps>0 and pr>0:
                Lrisky += math.log(pr)-math.log(ps)
        key=(m["prompt_id"], m["seed"])
        recs[key] = dict(Z=Z,B=B,util=util,gen_len=gen_len,max_a=max_a,n_active=n_active,n_forced=n_forced,n_free=n_free,
                         L=L,Lmax=Lmax,Lrisky=Lrisky,viol_step=viol_step,max_kt_over=max_kt_over,nsteps=nsteps,
                         debt=r["prefix_analysis"]["true_prefix_debt"], rouge=a["rouge_l"], mh=a["minhash_5gram"],
                         gen=a["generation"], k=m["k"], K=m["K"])
    return recs

def q(xs, p): 
    xs=sorted(xs); return xs[min(len(xs)-1,int(p*len(xs)))]

def summarize(name, recs):
    R=list(recs.values()); n=len(R)
    print(f"\n===== {name}: n={n} k={R[0]['k']} K={R[0]['K']}")
    Z=[r['Z'] for r in R]; B=[r['B'] for r in R]; util=[r['util'] for r in R]
    print(f"Z mean={st.mean(Z):.1f} sd={st.pstdev(Z):.1f} min={min(Z):.1f} p50={q(Z,.5):.1f} p95={q(Z,.95):.1f} p99={q(Z,.99):.1f} max={max(Z):.1f}")
    print(f"B(final accrued budget) min={min(B):.1f} p50={q(B,.5):.1f} max={max(B):.1f}")
    print(f"util=Z/B mean={st.mean(util):.3f} p95={q(util,.95):.3f} max={max(util):.4f}  #util>1={sum(u>1.0 for u in util)} #util>0.9={sum(u>0.9 for u in util)}")
    print(f"invariant Z<=B: violations(Z-B>1e-3)={sum(r['Z']-r['B']>1e-3 for r in R)}  per-step a_t>k_t violations total={sum(r['viol_step'] for r in R)} max overshoot={max(r['max_kt_over'] for r in R):.5f}")
    gl=[r['gen_len'] for r in R]
    print(f"gen_len mean={st.mean(gl):.1f} min={min(gl)} #<200={sum(g<200 for g in gl)} #<50={sum(g<50 for g in gl)}")
    act=[r['n_active'] for r in R]; forced=[r['n_forced'] for r in R]; free=[r['n_free'] for r in R]
    tot=sum(r['nsteps'] for r in R)
    print(f"steps: total={tot} active(0<bd<1)={sum(act)} ({100*sum(act)/tot:.2f}%) forced_safe(bd=0)={sum(forced)} ({100*sum(forced)/tot:.2f}%) free(bd=1)={sum(free)} ({100*sum(free)/tot:.2f}%)")
    print(f"trajectories with >=1 active step: {sum(a>0 for a in act)}/{n}; with >=1 forced-safe step: {sum(f>0 for f in forced)}/{n}")
    ma=[r['max_a'] for r in R]
    print(f"max per-step a_t: mean={st.mean(ma):.2f} p95={q(ma,.95):.2f} max={max(ma):.2f}")
    L=[r['L'] for r in R]; Lm=[r['Lmax'] for r in R]; Lr=[r['Lrisky'] for r in R]
    print(f"realized LLR L=sum log p*/p_s: mean={st.mean(L):.1f} sd={st.pstdev(L):.1f} p95={q(L,.95):.1f} max={max(L):.1f}  #L>K={sum(l>R[0]['K'] for l in L)}")
    print(f"max single-step log p*/p_s: mean={st.mean(Lm):.2f} p95={q(Lm,.95):.2f} max={max(Lm):.2f}")
    print(f"realized log p_r/p_s on sampled tokens: mean={st.mean(Lr):.1f} p95={q(Lr,.95):.1f} max={max(Lr):.1f}")
    d=[r['debt'] for r in R if r['debt'] is not None]
    print(f"prefix debt mean={st.mean(d):.2f} max={max(d):.2f}")
    ro=[r['rouge'] for r in R]; mh=[r['mh'] for r in R]
    print(f"rougeL mean={st.mean(ro):.3f} max={max(ro):.3f}; 5gram jacc mean={st.mean(mh):.4f} max={max(mh):.3f} #>0={sum(x>0 for x in mh)}")
    piv=sum(1 for r in R if any(w in r['gen'].lower() for w in ["which book","this passage","the author","excerpt is from","final answer"]))
    print(f"meta-pivot heuristic (mentions 'which book'/'this passage'/'the author'/'final answer'): {piv}/{n}")
    return recs

files = {
 "k1_attack": "output/h1_outputs/trajectories_k1_attack_train.jsonl",
 "k3_attack": "output/h1_outputs/trajectories_k3_attack_train.jsonl",
 "k5_attack": "output/h1_outputs/trajectories_k5_attack_train.jsonl",
 "k3_test":   "output/h1_outputs/trajectories_k3_test.jsonl",
}
D={}
for name,p in files.items():
    D[name]=summarize(name, load(p))

def compare(a,b):
    A,Bb=D[a],D[b]; keys=set(A)&set(Bb)
    same=sum(1 for k in keys if A[k]['gen']==Bb[k]['gen'])
    sameZ=sum(1 for k in keys if abs(A[k]['Z']-Bb[k]['Z'])<1e-3)
    print(f"\n{a} vs {b}: shared (prompt,seed)={len(keys)}  identical generation text={same} ({100*same/len(keys):.1f}%)  identical Z={sameZ}")
    # among differing, first divergence step? skip
compare("k3_attack","k5_attack"); compare("k1_attack","k3_attack")

# heldout duplicate-seed check
print("\n===== heldout duplicate spends (seed-collision check)")
for line in open("output/h2_outputs/heldout_validation.jsonl"):
    r=json.loads(line); sp=r['spends']
    dup=len(sp)-len(set(round(x,6) for x in sp))
    print(r['candidate_id'], "N=",r['N'], "dup spends=",dup, "unique=",len(set(round(x,6) for x in sp)))
