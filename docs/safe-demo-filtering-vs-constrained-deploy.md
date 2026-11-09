# Filter Demos vs Constrained Deploy — Research Brief

**For:** Harshitha / SAFE demonstrations  
**Domains:** navigation + tabletop IL under hard safety filters  
**Related longer note:** [`demo-filtering-vs-constrained-deploy.md`](./demo-filtering-vs-constrained-deploy.md) (experiment protocols, env stack)

---

## Executive recommendation

**Default (both nav and tabletop): train on original demos, deploy with a hard shield — then upgrade training so labels match the shield.**

| Domain | Default pipeline | Why |
|---|---|---|
| **Navigation** | **Train-then-constrain** (+ projected labels) | Filtering often deletes the only demonstrated passages near obstacles; shield freeze/chatter is easier to fix with projection + short recovery than with dead-ends |
| **Tabletop** | **Same default**, with **earlier use of oracle sequences** | Workspace/table shields turn “push into constraint” into progress stalls; single-step corrections rarely unstick grasps/approaches |

**Do this in order:**

1. **Ship B:** BC on raw demos + hard deploy shield (safety floor).  
2. **Upgrade to C:** train on shield-projected / executed labels (cuts proposal≠execution mismatch).  
3. **Filter-first (A) only when** illegal-demo rate is high *and* remaining safe demos still densely cover near-boundary states.  
4. **Add stuck detector + oracle recovery** in both regimes — shields guarantee safety, not progress.

**One-line answers**

1. Prefer **train-then-constrain** as default; **filter-then-train** when violations are frequent but safe coverage stays dense.  
2. Oracle help **yes** — prefer **short sequences** or **hold until the policy stops proposing the stuck action**; single one-step fixes are usually too weak.  
3. Publication: **workshop-ready as a bake-off**; **venue paper only** if framed as opposite failure modes + intervention design (not “we invented safe demos”).

---

## Tradeoff analysis

Assume: demos are mostly successful; deploy uses a hard action filter/shield (reject or project); training is BC/IL-family.

| Dimension | Filter-then-train (A) | Train-then-constrain (B) | Projected / shield-in-loop train (C) |
|---|---|---|---|
| **Distribution shift** | Train ≈ safe actions; missing states near constraints | Train includes illegal actions; deploy executes different ones | Train labels ≈ what deploy will execute |
| **Covariate / support shift** | **Primary risk:** eroded coverage near boundaries | Classic IL compounding + **shield-induced** next-state OOD | Lower proposal–execution shift; still compounding |
| **Safety violation rate** | Low if shield also on; demos already sanitized | ~0 under hard shield; **nominal** policy may still be unsafe if shield removed | Same as B with shield on |
| **Sample efficiency** | Worse when filtering discards rare corridor/grasp modes | Better use of all demos; wastes capacity on illegal modes | Best of both if projection is meaningful |
| **Compounding errors** | Drift into states with no demonstrated safe continuation → **dead-ends** | Drift + repeated illegal proposals → **freeze / oscillation** | Reduces freeze; dead-ends still possible if demos never showed recovery |
| **Task success under shield** | Fails when shortcuts were the only bridges | Fails when shield blocks the learned mode with no alternate | Usually strongest offline baseline |

**Mental model:** filtering fixes *what the policy proposes*; shielding fixes *what is allowed*. Neither invents recovery. Stuckness under a shield is often “safe but not progressing.”

---

## When each pipeline wins

### Filter-then-train wins when

- Illegal demo fraction is high (rough guide: ≳20–40%) **and**  
- Safe demos still cover approaches to obstacles / workspace limits / narrow gaps, and  
- You drop **illegal actions**, not whole trajectories that contain a few clips, unless those trajectories are mostly junk.

### Train-then-constrain wins when

- Filtering thins near-boundary density (narrow corridors, tight table clearances),  
- Constraints changed since demo collection (new margin, stricter CBF, new fixture), or  
- Unsafe demo actions were rare “glue” that connected otherwise safe segments.

### Prefer C (projected labels) over pure A or B when

- You can define a **least-restrictive** projection (nearest safe action / CBF-QP) that still makes progress,  
- You log `a_prop` and `a_exec` consistently in demos and rollouts.

**Kill the pure A-vs-B thesis early** if C ≥ B ≥ A everywhere with the obvious ranking — that result is useful engineering, weak as a flagship claim.

---

## Oracle intervention taxonomy

Use when stuck = no task progress for horizon *T*, or shield reject rate high / oscillation.

| Design | Benefit | Cost | Failure modes | Verdict |
|---|---|---|---|---|
| **V1: Single correction** | Cheap; fixes rare glitches | Low | Rarely leaves a basin; compounding error returns | Lower bound; expect weak on nav bottlenecks & tight tabletop |
| **V2: Repeat one action for *n* steps** | Simple; can shove through shallow minima; easy ablate *n* | Low–med | Wrong *n* or wrong direction creates new failures | Strong **cheap baseline** |
| **V3a: Repeat until π aligns *with* oracle** | Adaptive horizon; local “teach until consistent” | Med | Stubborn π → long takeover; agreement ≠ progress | Useful overwrite of a bad mode |
| **V3b: Repeat until π no longer proposes the stuck/illegal action** | Targets shield-rejection loops; often shorter than V3a | Med | Stops at vague “not that” with no good mode | **Best default adaptive rule** for freeze/oscillation |
| **V4: Oracle sequence** | Highest escape rate; adds intermediate states for BC | Highest | Expensive; overuse shifts dataset toward assisted recoveries | **Quality upper bound**; default for deep tabletop stalls |

**Recommendation:** V4 as quality ceiling; **V3b as cost/performance default** for shield loops; V2 as required baseline; V1 as negative control.

Map to interactive IL: V1–V2 ≈ short HG-DAgger gates; V3 ≈ gate until local policy agreement; V4 ≈ expert trajectorylets. Aggregate interventions into fine-tuning (mini-DAgger) separately from eval-only assist metrics.

---

## Related work pointers (searchable families)

Name lines carefully; pull exact bib entries when writing the paper. Do not claim these “solved” A vs B.

| Cluster | Anchors | Relevance |
|---|---|---|
| Interactive IL | DAgger (Ross et al.); HG-DAgger; SafeDAgger; EnsembleDAgger; LazyDAgger | Oracle interventions when learner visits bad states |
| Runtime shields / filters | Ames CBF line; Alshiekh et al. shielding for RL; backup CBF / MPC safety filters; RAIL (reachability-aided IL) | Deploy constraint enforcement; safety ≠ task success |
| Design-time safe IL | SAFE-GIL / MPC-SafeGIL (train-time safety vs test-time filters) | Closest “design vs deploy safety” narrative — different from offline demo sanitization |
| Constraint-aware IL | Racing / filter-in-training-objective work (e.g. Cao et al. constraint-aware IL) | Near Pipeline C |
| Offline safe IL / non-preferred demos | SafeDICE; related non-preferred-trajectory IL | Filtering/avoiding unsafe behavior offline — preference/cost angle |
| Demo curation | CoP-Miner / behavioral-fork prefix mining | Data-level safety complementary to runtime monitors |
| Recovery | Recovery RL; hierarchical task + recovery policies | Stuck under shield ≈ missing recovery, not missing BC loss |
| Offline / constrained RL | Action masking; CQL-family conservatism; CMDPs | Escalation if reward exists; train under same mask as deploy |
| Classic IL failure modes | Compounding error; covariate shift; causal confusion / copycat | Explains both pipelines’ closed-loop failures |

**Positioning:** contribution is a **comparative failure-mode study** (sanitize offline vs shield at deploy on fixed partially-illegal demos) **plus** an intervention design space for shield-induced deadlocks — not inventing shields or DAgger.

---

## Publication assessment

### Honest novelty bar

| Pitch | Likely outcome |
|---|---|
| “Filter unsafe demos → better BC” | Incremental hygiene |
| “BC + safety filter at deploy” | Known (RAIL / shielded LfD / SAFE-GIL comparisons) |
| “A vs B bake-off + opposite failure modes + budgeted intervention ablations” | **Solid robotics paper** if empirics are sharp and include Pipeline C |
| “Universal theorem that A or B always wins” | High risk; easy counterexamples |
| Lead with **V3b/V4 under shields** vs HG-DAgger / Recovery | Strongest *method* hook |

**Verdict:** as a standalone filter-vs-constrain dichotomy → **workshop / systems note**. As **“when sanitizing hurts, when shielding freezes, and how to intervene”** with surprising regime splits → **publishable at ICRA / IROS / CoRL-class** (harder at NeurIPS/ICML without theory or a crisp new algorithm). Nearby work (SAFE-GIL, RAIL, constraint-aware IL, CoP-Miner) already occupies the broad story — **narrow the claim**.

### Strongest angle

*Offline filtering and deploy-time shielding fix different mismatches. Filtering erodes support; shielding distorts executed actions. Interactive recovery with adaptive-horizon interventions closes the gap.*

Use A vs B as **motivating ablation**, not the title claim. Prefer leading with shield-compatible training (C) or deadlock interventions (V3b/V4).

### Scoop / incremental risks

- SAFE-GIL-style “train-time safety beats relying only on filters” reviews will appear.  
- Reviewers will ask: why not re-collect safe demos? why not train with the shield in the loop? why not HG-DAgger?  
- Predictable result (`C ≥ B ≥ A` except dense safe coverage) = incremental even if clean.

### What would make a strong paper

1. Clear opposite failure modes with diagnostics (coverage loss vs reject/oscillation).  
2. Constraint-severity sweep (% illegal demos).  
3. Pipeline **C** included (don’t false-dichotomize).  
4. Budget-matched V1–V4 curves; separate assisted vs autonomous success.  
5. Nav **and** tabletop; ideally one real-robot or high-fidelity check.  
6. Pre-registered decision rule for when A vs B vs C wins.

---

## Concrete next experiments

### Small (diagnostic, days)

- **S0 PointNav** with controllable margin gap (demo margin < deploy margin).  
- Run **A / B / C**; log success, stuck rate, shield reject rate, near-obstacle coverage before/after filter.  
- Add **V2 vs V4** under fixed oracle budget.  
- **Kill criterion:** if rankings are obvious everywhere, stop expanding the dichotomy; pivot to C or interventions.

### Medium (paper spine)

- **Safety-Gymnasium** Point/Car Goal (or F1TENTH if racing narrative) + **robosuite/robomimic or Meta-World** reach/push with table/workspace shield.  
- Illegal-demo rates {≈5%, 20%, 40%}; ≥5 seeds.  
- Full **V1–V4** budget curves; mini-DAgger fine-tune vs eval-only assist.  
- History ablation: progress/reject features + executed-action history (watch copycat stalls).

### Ambitious (venue upside)

- Real robot or high-fidelity tabletop with the same shield API.  
- Method claim: **disagree-until (V3b)** or **path-consistent / shield-in-loop BC** beating HG-DAgger + Recovery under matched budgets.  
- Optional: diffusion/ACT policies (masking/rejection may flip vs MLP-BC).  
- Optional theory sketch: when \(F(\mathcal{D})\) preserves approximate controllability on the safe set — only if you can state assumptions cleanly.

---

## Practical bottom line

1. **Deploy default:** train-then-constrain (B) → projected training (C); shield always on.  
2. **Filter-first** only after measuring coverage, not by default.  
3. **Nav and tabletop share the same default**; tabletop needs **sequences (V4)** sooner.  
4. **Oracle interventions help**; prefer V3b/V4 over single corrections.  
5. **Paper:** workshop if bake-off only; venue paper if failure-mode + intervention (or shield-compatible training) is the spine.  
6. Run the **small S0 pilot** before committing to a full paper on A vs B alone.

---

## Assumptions (flip these → flip the advice)

- Hard deploy shield exists and is enforceable on actions.  
- Demos are mostly successful, not pure failure noise.  
- Re-collecting fully safe demos is limited (otherwise just re-collect).  
- “Stuck” means no progress under the shield, not merely low reward.
