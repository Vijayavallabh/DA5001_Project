import json, math, statistics as st
for name,path,K in [("k3_attack","output/h1_outputs/trajectories_k3_attack_train.jsonl",600.0),("k3_test","output/h1_outputs/trajectories_k3_test.jsonl",600.0),("k1_attack","output/h1_outputs/trajectories_k1_attack_train.jsonl",200.0)]:
    S_s=[]; S_star=[]; first_safe=[]; 
    for line in open(path):
        r=json.loads(line); steps=r["per_step_log"]; gl=r["aggregate"]["generation_length_tokens"]
        ss=0.0; sp=0.0; nf=0; seen_free=False
        for s in steps[:gl]:
            ps,pst,bd=s.get("p_s_prob"),s.get("p_star_prob"),s.get("bd")
            if ps and ps>0: ss+=-math.log(ps)
            if pst and pst>0: sp+=-math.log(pst)
            if bd is not None and bd<=1e-6 and not seen_free: nf+=1
            else: seen_free=True
        S_s.append(ss); S_star.append(sp); first_safe.append(nf)
    n=len(S_s)
    vac=sum(1 for s in S_s if s <= K+math.log(2))
    cap=[min(1.0,(K+math.log(2))/s) if s>0 else 1.0 for s in S_s]
    print(f"{name}: n={n} K={K}")
    print(f"  safe-model surprisal of generated 200-tok continuation: mean={st.mean(S_s):.0f} p10={sorted(S_s)[n//10]:.0f} p50={sorted(S_s)[n//2]:.0f} p90={sorted(S_s)[9*n//10]:.0f}")
    print(f"  decoder surprisal of same text: mean={st.mean(S_star):.0f}")
    print(f"  event-bound cap (K+ln2)/S: vacuous (cap>=1) for {vac}/{n}; median cap={sorted(cap)[n//2]:.2f}")
    print(f"  leading forced-safe tokens per trajectory: mean={st.mean(first_safe):.2f} max={max(first_safe)}")
