# Demo Filtering vs Constrained Deployment for Imitation Learning

**Project:** SAFE demonstrations (Harshitha Rajaprakash)  
**Scope:** Navigation and tabletop manipulation policies trained from demonstrations, then deployed under safety filters / action constraints.  
**Question:** Should unsafe demonstration segments be filtered *before* training, or should the policy train on the original demos and rely on deploy-time safety filters? When the agent gets stuck, which oracle intervention designs help?

---

## 1. Verdict / Recommendation

### Primary recommendation (with conditions)

**Default for most navigation and tabletop setups: train on the original demonstrations, then deploy with a safety filter / shield — *and* plan for interactive recovery (oracle or recovery policy) when the shield induces stuck states.**

Filter-first is the better *offline* choice only when specific conditions hold (below). Pure filter-first without recovery is usually worse in practice because it creates unreachable or under-covered regions near constraint boundaries.

| Regime | Prefer | Why |
|---|---|---|
| Constraints are soft / rarely binding; demos almost always stay feasible | **Train-then-constrain** | Filter-first discards little signal and still risks cutting recovery-adjacent states; deploy filter is cheap insurance |
| Constraints are hard and demos frequently violate them, but feasible demos *still cover* the safe manifold well | **Filter-first (aggressive)** | Learning unsafe actions wastes capacity and teaches the policy to propose actions the shield will reject → freeze/oscillation |
| Constraints are hard *and* filtering leaves sparse coverage near boundaries / narrow corridors / tight grasps | **Train-then-constrain + recovery / interactive IL** | Filtered demos create dead-ends; full demos teach useful near-boundary behavior even if some actions are later masked |
| Constraints change between demo collection and deployment (new shield, stricter CBF, different table geometry) | **Train-then-constrain** (or re-collect) | Filter-first overfits yesterday’s feasible set; deploy-time shield adapts without refiltering the whole dataset |
| You can query an oracle online at train or fine-tune time | **Interactive IL (HG-DAgger-style) on top of either base** | Interventions fix the failure mode that pure offline filtering or pure shielding cannot: states the policy actually visits under the constrained rollout |

### Short answers to the three core questions

1. **Filter then train vs train then constrain:** Prefer **train-then-constrain** as the default, especially when filtering thins coverage. Prefer **filter-first** when demo violations are frequent *and* remaining safe demos still densely cover the deployable state space. In both cases, expect stuck states near the constraint boundary unless you add recovery or interactive correction.

2. **Oracle interventions when stuck:** Yes — interventions help, but **single one-step corrections are usually too weak** under compounding error. Prefer **short oracle sequences** or **hold intervention until the learner’s next action is consistent with the recovery direction**. Blindly repeating a single action for fixed `n` steps is a useful cheap baseline, not a complete solution.

3. **Publication-worthy?** As a *standalone* “filter vs shield” bake-off: **incremental**, likely a workshop / extended abstract / systems note unless the empirical picture is surprisingly sharp. As a **principled comparison of offline filtering vs deploy-time shielding under imitation + a designed intervention taxonomy for shield-induced stuckness**, with crisp failure-mode analysis and strong experiments: **publishable at a solid robotics / robot learning venue**, especially if framed as characterizing when each pipeline fails rather than claiming a universal winner.

---

## 2. Assumptions (flagged)

These assumptions bound the recommendations; violate them and the verdict can flip.

1. **Demonstrations are (mostly) successful task trajectories**, possibly with intermittent unsafe or constraint-violating actions (cutting corners, colliding lightly, illegal joint motions), not pure random or failed noise.
2. **The deploy-time safety mechanism is a filter/shield**: it rejects or projects illegal actions (action masking, CBF QP, control barrier / Lyapunov shield, geometric collision checker, table workspace clamp). It does **not** by itself provide a full recovery policy to a safe attractor unless specified.
3. **Training is behavioral cloning / IL family** (BC, dataset aggregation variants), not online RL from scratch. Offline RL with conservative objectives is related but secondary.
4. **Safety constraints at deploy are known and enforceable** on actions (or short-horizon action sequences). State constraints without a complete backup controller are harder.
5. **Oracle interventions are available in some budgeted form** for the “stuck” analysis (human teleop, scripted expert, privileged planner).
6. **“Stuck”** means: progress metric flat for a horizon, shield rejection rate high, oscillation between two actions, or freeze on a near-zero / null action — not merely low reward.
7. **Navigation and tabletop** are the target domains; conclusions may not transfer to contact-rich deformable manipulation or multi-agent traffic without re-checking coverage.

---

## 3. Theoretical framing

### 3.1 Two pipelines as different train/test mismatches

Define:

- \(\mathcal{D}\): raw demonstration dataset of state–action pairs \((s, a)\) (or segments).
- \(\mathcal{A}_{\text{safe}}(s)\): state-dependent safe action set enforced at deploy.
- Filter operator \(F\): keeps only pairs with \(a \in \mathcal{A}_{\text{safe}}(s)\) (and optionally drops entire segments that contain violations).
- Policy \(\pi_\theta(a \mid s)\) trained by imitation.
- Deployed policy: \(\tilde{\pi}(s) = \mathrm{Shield}(\pi_\theta(s), \mathcal{A}_{\text{safe}}(s))\).

**Pipeline A — Filter-first**

\[
\pi_A = \mathrm{IL}(F(\mathcal{D})), \quad \text{deploy } \tilde{\pi}_A = \mathrm{Shield}(\pi_A)
\]

Train distribution ≈ safe action conditional. Test dynamics still run under the shield. Main risk: **coverage / support mismatch** — states that appear only after unsafe demo transitions (or only near violations) are missing, so \(\pi_A\) is undertrained exactly where the shield binds.

**Pipeline B — Train-then-constrain**

\[
\pi_B = \mathrm{IL}(\mathcal{D}), \quad \text{deploy } \tilde{\pi}_B = \mathrm{Shield}(\pi_B)
\]

Train distribution includes illegal actions. At test time the shield **truncates or projects** the action, inducing:

- **Action-space truncation / covariate shift on actions:** the executed action \(a' = \mathrm{Shield}(a)\) may leave a next state outside the BC training support.
- **Proposal–execution mismatch:** \(\pi_B\) may put mass on illegal actions; the shield repeatedly rejects → freeze, dithering, or projection to a boundary action the learner never practiced as a *deliberate* action.

Both pipelines suffer classic IL issues (**compounding error**, Ross & Bagnell line of work) plus a constraint-specific failure mode: **shield-induced distribution shift**.

### 3.2 Distribution shift taxonomy (useful language for the paper)

| Shift type | Filter-first | Train-then-constrain |
|---|---|---|
| State covariate shift (IL compounding) | Yes | Yes |
| Missing mass near constraint boundary | **Primary failure** | Secondary |
| Action proposal ≠ executed action | Mild if \(\pi_A\) stays safe | **Primary failure** |
| Projection to boundary actions never demonstrated as intentional | Possible | Common |
| Dynamics shift from interrupted unsafe demos | If demos used unsafe shortcuts that change reachability | Same shortcuts unavailable → longer paths / new modes |

### 3.3 Why filtering can create dead-ends

Unsafe demo actions often serve as **bridges**: cut a corner, nudge an obstacle, use a slightly illegal joint configuration to escape a local basin. Filtering:

1. Removes the bridge actions.
2. May remove the **pre-bridge states** if you drop whole segments.
3. Leaves the policy in states whose only demonstrated continuations were filtered away → **absorbing near-failure sets** under the shield (local minima against a wall, grasp approaching singularity, robot facing a narrow gap it never practiced entering safely).

Formally, filtering can destroy **approximate controllability on the safe set** relative to the demonstrated task language: the safe set may still be controllable in principle, but not with the remaining demonstration support.

### 3.4 Why train-on-full then filter can freeze / oscillate

If \(\pi_B\) learned “go through the obstacle” as the mode:

1. Shield rejects \(a \notin \mathcal{A}_{\text{safe}}\).
2. Fallback may be: no-op, previous action, projection to nearest safe action, or sample until safe.
3. Nearest safe action may point *along* the obstacle (sliding) or *away* (retreat), neither of which was the demonstrated progress mode.
4. Next state remains similar → same illegal proposal → **oscillation** or **freeze**.

This is closely related to **safe action masking in RL/IL**: masking changes the effective policy class; without training under the mask (or correcting online), the learner optimizes the wrong conditional.

### 3.5 Connection to safety shields and constrained IL

- **Safety filters / shields / CBFs:** enforce hard constraints by modifying control inputs (Ames CBF line; Alshiekh et al. shielding for RL; related “shielded learning” literature). They guarantee constraint satisfaction *given a correct model/assumption*, not task progress.
- **Constrained imitation / safe IL:** learn policies that stay in safe sets (constraint penalties, safe BC, barrier-augmented objectives). Orthogonal to whether demos were prefiltered.
- **Offline RL with action masking:** train with masks in the loop so Q / policy never relies on illegal actions. The IL analogue is: **train BC with the shield in the loop** (project labels, or reject illegal labels) — a middle pipeline often better than pure A or B.

**Middle pipeline C (recommended baseline in experiments):**  
Train on demos but **relabel / project actions through the shield** during supervised learning, and/or drop only illegal *actions* while keeping states:

\[
a^\dagger = \mathrm{Proj}_{\mathcal{A}_{\text{safe}}(s)}(a), \quad \text{train on } (s, a^\dagger) \text{ when projection is well-defined}
\]

This reduces proposal–execution mismatch without discarding entire trajectories. It fails when projection is ambiguous or when the projected action is dynamically useless (normal force into a wall).

---

## 4. Related work clusters (searchable families; no fabricated citations)

Use these **concept / paper-family** anchors; pull exact bibliographic entries from the literature when writing the paper.

1. **Interactive imitation / dataset aggregation**  
   DAgger (Ross et al.); HG-DAgger / human-gated DAgger; related: SafeDAgger, LazyDAgger, Ensemble-DAgger. Core idea: query expert on states the *learner* visits to correct covariate shift.

2. **Safety filters, CBFs, shielding**  
   Control barrier functions (Ames et al.); shielding for RL (Alshiekh et al.); model predictive safety filters; reachable-set / Hamilton–Jacobi safety. Core idea: hard constraint enforcement at runtime, independent of how the nominal policy was trained.

3. **Safe / constrained imitation learning**  
   Safe BC variants; constraint-satisfying imitation; imitation with barrier certificates; learning from safe demonstrations only. Core idea: bake safety into the learning objective or demo selection.

4. **Action masking / constrained action spaces in offline RL**  
   Discrete/continuous action masks; conservatism (CQL family) as a soft cousin; policies trained under known invalid actions. Core idea: train under the same action restriction used at deploy.

5. **Recovery and safe backup policies**  
   Recovery RL; hierarchical safe RL (task policy + recovery policy); switching controllers when risk is high. Core idea: stuckness under a shield is often a missing *recovery* policy, not a missing BC loss term.

6. **Failure modes of imitation**  
   Compounding error; covariate shift; causal confusion; copycat learning. Core idea: explain why both pipelines degrade under closed-loop rollout.

7. **Learning from suboptimal / filtered demonstrations**  
   Ranking, preference, reweighted BC, filtering noisy demos. Useful when “unsafe” correlates with “suboptimal but informative.”

**Positioning vs these clusters:** the contribution is not inventing shields or DAgger. It is a **comparative systems + failure-mode study** at the intersection: *offline demo sanitization vs runtime shielding for IL*, plus an **intervention design space for shield-induced stuckness**.

### 4.1 Has this exact question been done before? (honest update)

**Short answer: the components and the design-time vs test-time dichotomy are well-trodden; a clean bake-off of “sanitize demos offline” vs “train on raw demos + deploy shield,” with shield-induced stuckness + intervention ablations, is not a famous solved landmark — but reviewers will correctly say nearby papers already occupy most of the conceptual space.**

Concrete nearby work (searchable; verify details in the papers themselves):

| Nearby work | What it already does | What it does *not* fully answer for this project |
|---|---|---|
| **SAFE-GIL** (train-time adversarial disturbance to collect safety-critical demos) | Explicitly contrasts **design-time** safety-aware IL vs **test-time safety filters**; shows filters can keep you safe while the task still fails / no recovery | Not a head-to-head of *offline filtering of illegal demo actions* vs deploy shield under a fixed demo set |
| **MPC-SafeGIL** | Same family; again argues train-time safety beats relying only on test-time filters; combines with filters | Same gap: data-sanitization vs shield, not disturbance-guided collection |
| **Constraint-aware IL for racing** (Cao et al., ~2025; assumes partially unsafe experts) | Closest relative: unsafe/unreliable demos + put a **safety filter into the BC training objective** (near our Pipeline C) | Focused on racing + training-with-filter method, not the filter-first dataset ablation + stuck-intervention taxonomy |
| **SECURE** (HRI 2024; CBF shielding for LfD/IRL) | Runtime CBF shielding for learning-from-demonstration | Not the offline demo-filter vs shield comparison |
| **In-distribution barrier filters** (Castaneda et al.) | Filter to stay near safe demo support | Assumes safe demos; different problem |
| **CoP-Miner / “behavioral fork” data curation** | Offline **data-level** removal of unsafe *suffixes* of failed trajectories; explicitly positions data interventions as complementary to runtime monitors | About failure suffixes / paired success–failure logs, not hard deploy constraints that successful demos partially violate |
| **Path-consistent safety filtering for diffusion policies (PACS)** | Notes that naive runtime filters **change actions in ways unseen in training** (exactly our proposal–execution mismatch) | Method for better filtering under diffusion policies, not the A vs B dataset question |
| **Recovery RL / DFR / Rewind-IL / HG-DAgger** | Recovery, respawn, or human interventions when failing or drifting | Do not settle filter-first vs train-then-constrain |

**Implication for novelty:** You should **not** pitch “nobody has thought about safe demos vs safety filters.” Pitch: *under a fixed set of demos that partially violate known deploy constraints, when does offline sanitization help or hurt relative to training on raw labels and shielding at deploy — and how should oracle interventions unstick shield-induced deadlocks?* That narrower question is still crowded, but not identical to SAFE-GIL or Cao et al.

---

## 5. Oracle intervention design space

When the agent is stuck under either pipeline, interventions differ in **query cost**, **bias injected into the dataset**, and **alignment with DAgger-style theory**.

### 5.1 Variant comparison

| Variant | Description | Pros | Cons | When useful |
|---|---|---|---|---|
| **V1: Single correction** | One oracle action (or short teleop blip), then return control | Cheap; low annotator burden; good for rare glitches | Often insufficient under compounding error; may not leave the stuck basin | Mild stuckness; high-frequency human oversight; already near a good manifold |
| **V2: Repeat single intervention for `n` steps** | Hold the same oracle action for fixed horizon `n` | Simple; can push through shallow local minima; easy to ablate `n` | Wrong direction held too long → new failures; `n` is brittle across scenes | Navigation past thin obstacles; tabletop “nudge out of collision” with calibrated `n` |
| **V3a: Repeat until learner aligns *with* intervention** | Keep applying oracle action until \(\pi_\theta\)’s argmax / sample agrees with the intervention (within tolerance) | Adaptive horizon; stops when local policy is consistent; DAgger-like “teach until learned locally” | Can run long if policy is stubborn; agreement ≠ task progress; threshold tuning | When stuckness is caused by persistent illegal mode that needs overwriting |
| **V3b: Repeat until learner aligns *against* original stuck action** | Hold recovery until \(\pi_\theta\) no longer proposes the bad action (even if not matching oracle exactly) | Directly targets the failure mode (stop proposing the shield-rejected action); often shorter than full alignment | May stop in a vague “not that” policy that still has no good mode | Freeze/oscillation against a hard constraint; clear identifiable bad action |
| **V4: Oracle action *sequence*** | Query a multi-step recovery (open-loop snippet or short teleop segment) | Highest chance to escape basins; provides intermediate states for BC; matches how humans unstick robots | Most expensive; sequence distribution shift if overused; needs clear termination | Narrow passages; multi-step regrasps; any deep local minimum |

### 5.2 Connection to DAgger-style querying

- **DAgger** queries the expert on states visited by the learner and aggregates; theoretically reduces to no-regret under certain assumptions.
- **HG-DAgger** lets the human decide when to intervene (gated), which matches “intervene when stuck / unsafe” better than querying every state.
- Mapping:
  - **V1** ≈ minimal HG intervention (often too short for theory’s benefit).
  - **V2** ≈ fixed-horizon gated intervention (practical heuristic).
  - **V3a/b** ≈ **closed-loop gating until local policy agreement** — a hybrid of HG-DAgger and “intervene until safe/confident.”
  - **V4** ≈ providing expert *trajectorylets*, closer to daggering on a short recovered path.

**Recommendation for intervention policy:**  
Use **V4 (short sequences)** as the quality upper bound; use **V3b** as the best cost/performance default for shield rejection loops; use **V2** as a strong simple baseline; treat **V1** as a lower bound that will often fail in navigation bottlenecks and tight tabletop geometries.

### 5.3 What to log for each intervention

For scientific comparison, log: stuck trigger reason; shield rejection rate; oracle horizon; whether \(\pi\) flipped; post-intervention success; states added to the aggregate dataset; and whether interventions are used only at eval or also for fine-tuning (offline eval-only interventions measure *assistability*, not autonomous improvement).

---

## 6. Stuck-state analysis

### 6.1 Filter-first → dead-ends

**Mechanism:** support erosion near constraints.  
**Symptoms:** policy hesitates at corridor entrances; approaches table edge then stops; cycles in free space away from the goal because goal-directed demos took slight constraint violations as shortcuts.  
**Aggravators:** segment-level dropping; scarce demos; narrow \(\mathcal{A}_{\text{safe}}\); multimodal expert (left vs right around obstacle) where one mode was mostly unsafe and removed.

### 6.2 Train-then-constrain → OOD executed actions / freeze / oscillation

**Mechanism:** proposal–execution mismatch + projection to undemonstrated boundary actions.  
**Symptoms:** high shield reject rate; chatter at constraint boundary; null-action freeze; sliding along an obstacle without progress; gripper closing into an object while Cartesian shield projects motion away.  
**Aggravators:** demos that heavily rely on illegal shortcuts; naive shield fallback (zero action); no training-time projection.

### 6.3 Shared failure: compounding error into the bad set

Even a perfect safe demo distribution does not prevent drift into states with no demonstrated recovery. Shields keep you *safe* while stuck; they do not create *progress*. Hence recovery RL / interactive IL remain relevant under both pipelines.

### 6.4 Diagnostic checklist (practical)

1. Measure **fraction of demo actions illegal under deploy shield**.  
2. Measure **state-space coverage loss** after filtering (e.g., proxy via embedding kNN density, or visitation histograms in navigation maps / end-effector voxels).  
3. At rollout: **shield reject rate**, **time-to-first-reject**, **progress before stuck**, **oscillation detector**.  
4. Ablate shield fallback: zero vs previous vs projection vs sample-until-safe.  
5. Compare Pipeline C (train with projected labels) against A and B.

---

## 7. Publication assessment (honest)

### 7.1 Novelty bar

| Claim shape | Likely reception |
|---|---|
| “We filter unsafe demos and BC works better” | Incremental; many labs already do this as hygiene |
| “We deploy a safety filter with BC” | Incremental; shields + IL is known |
| “We systematically compare filter-first vs train-then-shield for IL, show opposite failure modes, and evaluate intervention variants for stuckness” | **Credible robotics contribution** if experiments are thorough |
| “We give a theorem that one always dominates” | High risk; reviewers will demand unrealistic assumptions or find counterexamples |
| “New intervention rule (agree-until / disagree-until) for shielded IL + analysis” | Potentially the sharpest novel *method* hook if it clearly beats HG-DAgger baselines |

**Honest take:** this has **partially been done** in spirit. Design-time vs test-time safety for IL is an active thread (SAFE-GIL / MPC-SafeGIL; constraint-aware IL with filters in the training loop; CBF-shielded LfD; data curation that strips unsafe suffixes). Claiming the broad dichotomy as novel will get rejected. Publishability hinges on a **narrower empirical contribution**: fixed partially-illegal demos + known deploy constraints + opposite failure modes of sanitize-vs-shield + budgeted intervention ablations for stuckness — ideally with a result that surprises relative to “just use SAFE-GIL / just train with the filter.” Without that, expect workshop / incremental robotics paper territory, not a clean slate discovery.

### 7.2 Suggested framing / angle

**Working title angles (pick one spine):**

1. *Safe After Learning or Safe Before Learning? Failure Modes of Demonstration Filtering vs Runtime Shields for Imitation.*  
2. *Unsticking Shielded Imitation: An Intervention Design Space for Constraint-Induced Deadlocks.*  
3. *Train on Unsafe Demonstrations, Deploy with Filters: When Does Sanitizing the Dataset Help?*

Recommended spine: **(1) as the main empirical paper**, with **(2) as the method section** that elevates novelty.

**Thesis statement to defend:**  
*Offline filtering and deploy-time shielding address different mismatches; neither alone ensures task success under hard constraints. Filtering fails by eroding support; shielding fails by distorting executed actions. Interactive recovery with adaptive-horizon interventions closes the gap.*

### 7.3 Experiments needed (minimum credible set)

1. **Two domains:** grid/continuous navigation with obstacle CBFs or occupancy shields; tabletop reach/grasp with workspace + collision shields.  
2. **Three training pipelines:** A filter-first, B train-then-shield, C projected-label training.  
3. **Constraint severity sweep:** % illegal demo actions ∈ {5%, 20%, 40%+}.  
4. **Coverage metrics** + success + shield reject rate + stuck rate.  
5. **Intervention ablations:** V1–V4 under a fixed oracle budget (actions or seconds).  
6. **Autonomous vs assisted:** interventions for evaluation-only assist vs interventions aggregated for fine-tuning (mini-DAgger).  
7. **Baselines:** plain BC; BC + shield; HG-DAgger; Recovery-style backup if feasible.  
8. **Negative result reporting:** show regimes where each pipeline wins — reviewers distrust universal claims.

### 7.4 Venues and what reviewers will attack

| Venue type | Fit | Likely attacks |
|---|---|---|
| CoRL / RSS / ICRA / IROS | Best fit | “Is the shield realistic?”, “toy constraints”, “why not just collect safe demos?”, “missing real robot” |
| CoRL / RSS (robot learning emphasis) | Strong if intervention story is crisp | “novelty vs HG-DAgger + shielding”, demand real-robot or high-fidelity sim |
| NeurIPS / ICML / ICLR | Harder | “insufficient theory”, “incremental empiricism”, “narrow robotics engineering” |
| Workshops (safe RL, robot learning, IL) | Excellent first outing | Lower bar; good for feedback before full paper |

**Preempt reviewer attacks:**

- Define the shield precisely; release constraint implementations.  
- Include Pipeline C so you are not falsely dichotomizing.  
- Separate **safety** (constraint violations) from **success** (task completion).  
- Budget-match oracle methods.  
- Avoid claiming novelty for DAgger or CBFs; claim novelty for the **comparative analysis + intervention taxonomy under shields**.  
- Discuss when re-collecting safe demos dominates both pipelines (often true industrially — address it head-on).

---

## 8. Concrete experimental protocol

### 8.1 Shared setup

- **Expert:** scripted planner or human teleop that *can* violate constraints (aggressive demos).  
- **Shield:** deterministic, same at eval for all methods (document fallback).  
- **Policy:** same architecture (e.g., BC on states → actions; optional diffusion BC if that is the lab standard — keep architecture fixed across A/B/C).  
- **Metrics:** success rate; constraint violation count (should be ~0 with hard shield); stuck rate; mean completion time; shield reject rate; oracle action budget (for intervention arms).  
- **Seeds:** ≥5; report mean ± CI.  
- **Assumption to encode in code:** illegal actions are detectable online via the same shield used at deploy.

### 8.2 Navigation protocol

1. **Env:** 2D or SE(2) navigation among obstacles; goal-conditioned. Optional partial observability later — start fully observed.  
2. **Shield:** collision checker on candidate \(\Delta x, \Delta y, \Delta\theta\) or CBF-QP on simple robot dynamics.  
3. **Demo generation:** expert with shortened paths that clip obstacle margins at rates {low, mid, high}.  
4. **Filter-first:** drop illegal transitions; optional variant drops any trajectory with >k violations.  
5. **Train-then-constrain:** BC on all; deploy with shield + chosen fallback.  
6. **Projected training:** replace illegal actions with projected safe actions when the projection retains ≥τ progress toward expert’s next state; else drop pair only.  
7. **Stuck definition:** no progress in geodesic-to-goal for T steps *or* reject rate >ρ over a window.  
8. **Interventions:** on stuck, apply V1–V4 with equal max oracle budget B; continue episode; optionally aggregate and fine-tune for K rounds.  
9. **Plots that matter:** success vs % illegal demos; stuck rate vs coverage proxy; oracle budget–success curves per intervention variant.

### 8.3 Tabletop protocol

1. **Env:** reach / push / pick-and-place with workspace limits and self-collision / table collision shield.  
2. **Shield:** clamp EE pose to workspace; reject colliding Jacobian motions; optional finger joint limits.  
3. **Demos:** teleop or scripted that occasionally scrape table or exit soft workspace (common in real teleop).  
4. Same A/B/C pipelines as navigation.  
5. **Stuck definition:** EE speed below ε and grasp not progressed for T steps, or repeated shield rejects during approach.  
6. **Interventions:** especially V4 for regrasp-like recoveries; V3b for “stop pushing into the table.”  
7. **Extra metric:** fraction of episodes needing intervention; post-intervention autonomous success without further help.

### 8.4 Decision rule to report (pre-register)

- If filtering removes >X% of transitions *or* reduces local density near obstacles/workspace boundary below Y: predict **B or C wins**.  
- If illegal demo rate is high but safe demos remain dense: predict **A wins**.  
- If shield reject rate under B exceeds Z with low success: predict **interventions or C required**.  
Choose X/Y/Z from a pilot split; state them in the paper to avoid post-hoc storytelling.

### 8.5 Simulation scenarios I would actually run

Design principle: **one cheap diagnostic env (fast iteration) + one credible nav env + one tabletop env.** Keep the shield API identical across all three so A/B/C and interventions are apples-to-apples.

#### Scenario S0 — PointNav corridors (diagnostic, week 1)

**Purpose:** isolate filter-induced dead-ends vs shield-induced freeze without vision or contact complexity.

| Piece | Choice |
|---|---|
| Env | 2D continuous Point / Dubins car in a maze or random convex obstacles (custom Gymnasium, or strip down Safety-Gymnasium Point-Goal) |
| State | \((x,y,\theta)\) or \((x,y,v,\omega)\) + goal relative pose; optional lidar |
| Action | \((\Delta x, \Delta y)\) or \((v, \omega)\) |
| Task | Reach goal without collision |
| Shield | Reject / project any action whose short rollout intersects inflated obstacles (disk robot + margin \(m\)) |
| Expert | A* / RRT* on a **smaller** margin \(m_{\text{demo}} < m_{\text{deploy}}\), or noisy teleop that clips corners |
| Illegal-demo knob | Sweep \(m_{\text{deploy}} - m_{\text{demo}} \in \{0, \varepsilon, 2\varepsilon, \ldots\}\) → controls % illegal actions without hand-labeling |
| Stuck | No decrease in geodesic-to-goal for \(T\) steps, or shield reject rate \(>\rho\) over a window |
| Why first | Hours to implement; clearest coverage histograms; proves instrumentation |

**Expected qualitative picture:** large margin gap → filter-first drops corridor-entry states → dead-ends; train-then-shield → agent presses into walls and chatters.

#### Scenario S1 — Safe navigation (credible nav)

**Purpose:** paper-facing nav result with a known safe-RL stack.

| Piece | Choice |
|---|---|
| Env | **Safety-Gymnasium** `SafetyPointGoal1-v0` / `SafetyCarGoal1-v0` (hazards + pillars); keep layout fixed across seeds for fair coverage metrics |
| Obs | Start with **state vector**; add vision only in a later appendix |
| Shield | Hard collision / hazard checker on candidate action (or simple CBF-QP on Point dynamics). Prefer a **least-restrictive** projection to nearest safe action when feasible |
| Expert | Scripted goal-seeker with reduced hazard inflation, **or** privileged MPC that occasionally violates deploy hazards; collect \(N \in \{50,200,500\}\) demos |
| Variants | (i) narrow gap between hazards, (ii) open field with sparse hazards — tests coverage sensitivity |
| History | Ablate Markov vs obs stack vs executed-action history vs reject-bit (§10) |

**Do not** start with Ant/Doggo — dynamics noise obscures the A/B/C signal.

#### Scenario S2 — Tabletop reach / push (manipulation)

**Purpose:** workspace + table-collision shields; stall looks like “pushing into the table.”

| Piece | Choice |
|---|---|
| Env (pick one stack and stick) | **MuJoCo** Franka / UR5 reach-push (Meta-World `reach`/`push` as base), **or** Robosuite Lift/Stack, **or** LeRobot-style SO-101 tabletop MuJoCo if that matches hardware plans |
| Task ladder | (2a) EE reach to target above table → (2b) push cube to goal → (2c) pick-place only if (2a–b) are clean |
| Action | Delta EE pose (+ gripper for 2c); keep dim modest |
| Shield | (i) EE \(z \ge z_{\min}\) (table plane), (ii) axis-aligned workspace box, (iii) optional self-collision reject via MuJoCo contact on non-task geoms |
| Expert | IK teleop or scripted waypoints that **intentionally** use \(z_{\min}^{\text{demo}} < z_{\min}^{\text{deploy}}\) (scrape / deep approach), and/or slight workspace overshoot |
| Stuck | \(\|\dot{p}_{\text{EE}}\| < \varepsilon\) and \(\Delta\) task error \(< \delta\) improvement for \(T\) steps, **or** repeated \(z\)-shield rejects |
| Intervention oracle | Privileged scripted recovery: lift \(+z\), retreat in \(xy\), then resume approach (natural V4 sequence) |

**Why this stall is perfect for the project:** shield keeps you “safe” while the cube never moves — success and safety diverge cleanly.

#### Scenario S3 (optional stretch) — Narrow-passage nav or door/slot

Only after S0–S2: a single homotopy bottleneck (gap width ≈ robot diameter). Filter-first often deletes the only demonstrated passage mode; train-then-shield oscillates at the mouth of the gap. High illustrative value, easy for reviewers to grasp visually.

### 8.6 How to set it up (lab recipe)

#### A. Unified experiment API (implement once)

```text
demo = (obs_t, a_prop_t, a_exec_t, shield_reject_t, info_t)
shield(obs, a_prop) -> a_exec, reject_flag
stuck(window) -> bool
oracle.intervene(variant, budget) -> actions
```

Every pipeline (A/B/C) and every env must use the **same** `shield` and logging schema. Store both proposed and executed actions from day one (needed for history + Pipeline C).

#### B. Demo generation (controllable illegality)

1. Fix deploy shield parameters \(\mathcal{C}_{\text{deploy}}\).  
2. Generate expert with relaxed constraints \(\mathcal{C}_{\text{demo}} \subsetneq \mathcal{C}_{\text{deploy}}\) (smaller obstacle inflation, lower table plane, larger workspace).  
3. Report **illegal action rate** = fraction of demo actions with `shield(obs, a_prop).reject` under deploy shield. Target sweep **{5%, 20%, 40%}**.  
4. Hold task success of demos high (only successful or near-successful trajectories) so you study *constraint mismatch*, not failed teleop noise.  
5. Optional second axis: demo count \(N\), to see data-scarcity interaction with filtering.

#### C. Pipelines to run on each scenario

| ID | Train data | Deploy |
|---|---|---|
| A | Drop rejected transitions (and optionally whole traj if reject rate high) | Shield on |
| B | All raw demo actions | Shield on |
| C | Relabel \(a \leftarrow a_{\text{exec}}\) / projection when well-defined | Shield on |
| B-hist / C-hist | Same + obs/action/reject history | Shield on |
| +Ix | Any of the above + intervention variant Vx on stuck, budget-matched | Shield on |

Baselines worth 1–2 days each: plain BC no shield (unsafe upper bound on task speed); HG-DAgger-style human/script gate if oracle exists.

#### D. Training defaults (keep boring)

- Policy: MLP on state for S0/S1; small CNN/MLP or your lab’s ACT/diffusion BC for S2 — **freeze architecture across A/B/C**.  
- Loss: MSE / Huber on actions (or chunked MSE).  
- Train until val action error plateaus; evaluate closed-loop, not val MSE.  
- Seeds: ≥5; same demo set across methods per seed family.

#### E. Metrics dashboard (must-have plots)

1. Success vs illegal-demo rate (A vs B vs C).  
2. Stuck rate + mean shield reject rate.  
3. Coverage proxy: histogram of demos near constraint boundary (distance-to-obstacle / \(z - z_{\min}\)) before vs after filtering.  
4. For interventions: success vs oracle action budget for V1–V4.  
5. Stall re-entry count with vs without history.

#### F. Suggested calendar (practical, not calendar-padded)

1. **S0 + logging + shield + A/B/C** until plots look sane.  
2. Add stuck detector + V2 and V4 interventions on S0.  
3. Port API to **S1 Safety-Gymnasium Point/Car Goal**.  
4. Build **S2 tabletop** with \(z\) + workspace shield; scripted scrape demos.  
5. History ablations and kill-criteria check (§12): if rankings are obvious everywhere, **stop expanding the dichotomy** and pivot method focus.

#### G. Stack suggestions (minimize yak-shaving)

| Need | Lean default |
|---|---|
| Nav benchmark | `safety-gymnasium` + MuJoCo |
| Fast 2D prototype | Custom Gymnasium PointEnv (100–200 LOC) |
| Tabletop | MuJoCo + existing Franka/Meta-World/Robosuite env you already know; or LeRobot SO-101 MuJoCo if hardware-aligned |
| Orchestration | Single `configs/` for shield margins, demo illegality, seeds |
| Viz | Top-down nav overlays of rejects; tabletop side view of \(z\) scrapes |

#### H. Assumptions for this setup

- Sim shield ≈ intended deploy constraint family (margin/table/workspace).  
- Expert can be scripted; human teleop optional later.  
- Vision deferred until state-based story is solid (vision makes shield state-estimation a confound — SAFE-GIL already stresses that).

### 8.7 Open-source environments used in related work (what to actually install)

Yes — there are solid open-source options. Prefer **community baselines the related papers already sit next to**, not a brand-new custom world (unless you need S0 for debugging).

#### Recommended shortlist (start here)

| Priority | Environment | Why it matches this project | Used as / near baseline in |
|---|---|---|---|
| **1 — Nav** | **[Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)** (`SafetyPointGoal*`, `SafetyCarGoal*`) | Native hazard/obstacle costs, MuJoCo, Gymnasium API; easiest place to attach a hard shield and report constraint metrics | Standard **safe RL** successor to OpenAI Safety-Gym (NeurIPS’23 benchmark family). Closest “official” open safe-nav suite |
| **2 — Racing / constraint-aware IL** | **[F1TENTH Gym](https://github.com/f1tenth/f1tenth_gym)** (+ IL repos such as `mlab-upenn/f1tenth_il`, SafeIL-style codebases) | Track limits = natural hard constraints; demos often push limits → illegal-vs-deploy mismatch is realistic | Directly adjacent to **constraint-aware IL for autonomous racing**; F1TENTH is the community racing baseline |
| **3 — Tabletop IL** | **[robosuite](https://github.com/ARISE-Initiative/robosuite) + [robomimic](https://github.com/ARISE-Initiative/robomimic)** (Lift / Can / Square) | Standard BC datasets + teleop; add workspace / table / collision shield yourself | De-facto **manipulation IL** baseline (CoRL robomimic study); CoP-Miner-style data curation also uses RoboMimic Can |
| **Alt tabletop** | **[Meta-World](https://github.com/Farama-Foundation/Metaworld)** (`reach`, `push`, `pick-place`) | Lighter than robosuite; good for S2 reach/push ladder | Ubiquitous continuous-control / multi-task manipulation baseline (less “safe IL” specific) |

#### Also open-source, but second-line for *your* question

| Environment | Role | Caveat |
|---|---|---|
| **MetaDrive** | Driving + procedural maps; used in safe/generalization driving work | Heavier than PointNav; good if you want driving narrative |
| **highway-env** | Lightweight lane/collision driving | Very simple dynamics; fine for a toy, weak as sole paper env |
| **Gymnasium-Robotics / Minari** (ex-D4RL migration) | Offline datasets + robotics envs | Strong for offline RL comparisons; safety shield is not built-in |
| **LIBERO / LIBERO-Safety** | Language-conditioned manipulation + safety suites | Excellent if you go VLA/vision-language; overkill for first BC+shield study |
| **X-Plane taxiing** (used in SAFE-GIL) | Vision taxi benchmark in that paper | Not a lightweight open research gym; painful setup vs Safety-Gymnasium |
| **Custom 2D Point/unicycle** (SAFE-GIL-style ground nav) | Many safe-IL papers use a **simple custom** obstacle field | Fine as S0 diagnostic; reviewers will still want one *named* baseline (Safety-Gymnasium or F1TENTH) |

#### What related-work clusters actually ran (honest map)

- **SAFE-GIL / MPC-SafeGIL:** custom 2D ground nav + X-Plane taxi + Crazyflie hardware — *not* primarily Safety-Gymnasium. So “SAFE-GIL env” ≠ one pip package; for open reproducibility, **Safety-Gymnasium is the community substitute** for their nav setting.  
- **Constraint-aware IL (racing):** F1TENTH-style racing sims.  
- **CBF-shielded LfD / safe IRL:** often custom racecar/drone or driving sims; SECURE reports robotic + driving sims + real knife-cutting.  
- **Manipulation IL / demo filtering:** **robomimic + robosuite** (Lift/Can/Square) is the citation everyone recognizes.  
- **Safe RL methods papers:** **Safety-Gymnasium** (and legacy Safety-Gym) is the citation magnet.

#### Concrete “install these three” plan

1. `safety-gymnasium` → `SafetyPointGoal1-v0` / `SafetyCarGoal1-v0` for nav + hazards.  
2. `f1tenth_gym` if you want a constraint-aware-IL-comparable racing story (optional but high relevance).  
3. `robosuite` + `robomimic` Lift/Can (low-dim first) for tabletop; wrap a table/workspace shield outside the env.

You still implement the **shield wrapper** yourself in all cases — almost none of these ship “demo filtering vs deploy shield” as a ready benchmark. What they give you is **credibility + demos/API**, not the experimental question pre-solved.

#### Minimal starter commands (illustrative)

```bash
pip install safety-gymnasium
# then: gymnasium.make("SafetyPointGoal1-v0")

# tabletop IL stack (versions matter — follow current robomimic docs)
# robosuite + robomimic from ARISE repos; download Lift/Can PH low-dim datasets
```

---

## 9. Would AIRL / richer RL help instead of BC?

### Short answer

**Sometimes yes for task progress under constraints — not as a free substitute for the filter/shield question, and often not worth it as the first lever.**  
AIRL and stronger RL can *in principle* invent safe alternatives that BC cannot copy, but they bring reward ambiguity, sample cost, and the same shield mismatch unless you train under the constraint. Prefer **BC → projected/shield-in-the-loop BC → offline RL with action masks → online constrained/safe RL** as an escalation ladder, not a jump straight to AIRL.

### What BC gets wrong that IRL/RL might fix

| BC failure under constraints | Can AIRL / RL help? | How |
|---|---|---|
| Copies illegal demo actions → shield rejects / freeze | **Partially** | Reward-seeking policy may prefer other actions if illegal ones are low-return or masked |
| Missing safe bridges after filtering (dead-ends) | **Yes, if** online (or richly covered offline) exploration finds detours | Needs env interaction or diverse offline support |
| Compounding error / covariate shift | **Partially** | Online RL / DAgger-like interaction fixes visitation; pure offline AIRL less so |
| Proposal ≠ shielded execution | **Only if** trained with the shield/mask in the loop | Unconstrained AIRL + deploy shield can still freeze |
| Cannot exceed suboptimal expert | **Yes (IRL/RL’s main pitch)** | Useful when demos are aggressive/unsafe *and* a better safe policy exists |

### Method-by-method reality check

1. **AIRL / GAIL / adversarial IRL family**  
   - Learns a reward (or occupancy discriminator) then RL. Can generalize beyond demonstrated actions.  
   - **Fails to magically encode hard deploy constraints** unless constraints enter the MDP (cost, absorption, shield during rollouts) or the reward is shaped against violations.  
   - If demos contain illegal shortcuts that look “expert-like,” AIRL may **infer a reward that likes the shortcut**, then the deploy shield fights the optimized policy — same proposal–execution mismatch as BC, sometimes worse because the policy is more confidently illegal.  
   - Unstable, hyperparameter-heavy, weak on high-dim vision tabletop without careful engineering. Exists related work (e.g. CBF-augmented IRL / safe IRL) showing you still bolt safety on.

2. **Offline RL (CQL / IQL / diffusion QL, etc.)**  
   - Better than BC when demos are mixed quality and you have a reward (or can define sparse task success).  
   - With **action masking / constrained action sets in training**, this is one of the cleanest upgrades over BC for train-then-constrain.  
   - Without masks: can still stitch unsafe transitions. Filtering demos then offline RL = filter-first again (coverage issues).  
   - Needs a reward signal BC doesn’t; for pure imitation setups that’s an extra assumption.

3. **Online RL / constrained RL / Recovery RL**  
   - Strongest at escaping shield-induced stuckness *if* you can reset safely and afford interaction.  
   - Train **with the shield on** (shielded RL) so the policy never relies on rejected actions.  
   - This largely **sidesteps** “filter demos vs deploy shield” by changing the learning paradigm — demos become optional warm-starts, not the whole algorithm.  
   - Cost: sim fidelity, real-robot sample cost, constraint specification, and you still need a task reward.

4. **Inverse constraint RL / constraint inference from demos**  
   - Different goal: infer the *constraint*, not just a policy (NeurIPS-style constraint inference / ICRL line).  
   - Useful if the deploy shield is unknown; **less relevant** when you already have an explicit safety filter (your setting).

### Does switching away from BC retire the original research question?

**No.** The core mismatch is:

- demos that disagree with \(\mathcal{A}_{\text{safe}}\), plus  
- a deploy mechanism that changes executed actions.

Any learner that (a) fits demo-like behavior and (b) is deployed under a hard filter can freeze or oscillate. AIRL changes *how* the nominal policy is produced; it does not remove the need to either **sanitize labels**, **train under the shield**, or **recover when stuck**.

What *does* change: with online constrained RL, “filter-first vs train-then-constrain” becomes less central than “**was the shield in the training loop?**” and “**is there a recovery objective?**”

### Recommendation ladder (practical)

1. Stay on **BC + Pipeline C (projected / shield-in-loop labels)** unless demos are clearly multi-quality or you have a reliable reward.  
2. If demos are suboptimal/unsafe but you have reward + decent offline coverage: try **offline RL with action masking under \(\mathcal{A}_{\text{safe}}\)** — usually a better next step than AIRL.  
3. Reach for **AIRL/GAIL** only when you need reward transfer / to beat the expert *and* you can run RL under the shield (or with violation costs). Treat unconstrained AIRL + late shield as a likely failure mode.  
4. If interaction is allowed: **shielded / constrained RL ± recovery policy**, optionally BC/AIRL-initialized — this is the principled way to “not get stuck,” not a fancier imitator alone.

### Assumption flags

- Assumes hard constraints are known at train or deploy (true for your shield setting).  
- Assumes tabletop/nav may be vision-based — adversarial IRL is much harder there than state-based nav.  
- Assumes “help” means higher autonomous success under the same shield, not just nicer offline imitation loss.

---

## 10. Action / state history for BC (avoiding stall)

### Short answer

**Yes — history is useful for BC, and it is one of the cheapest anti-stall levers — but action history alone will not invent an escape if demos never show “stuck → recover,” and naive previous-action inputs can worsen stalls via copycat behavior.**  
Prefer **state / progress history + executed (post-shield) actions**, and consider **action chunking**; treat raw “last commanded action” as optional and carefully ablated.

### Why history can prevent re-entering a stall

Markov BC on \(s_t\) only cannot distinguish:

- first approach to an obstacle (still recoverable), vs  
- fifth rejected shove into the same constraint (stall / oscillation).

Conditioning on a short window lets the policy learn patterns like: *recent motion ≈ 0*, *recent executed actions were projections*, *goal distance flat* → try a different mode (strafe, reverse, regrasp) **if those alternatives appear in the data**.

### What to feed (ranked)

| Signal | Helps stall? | Notes |
|---|---|---|
| **State / observation history** (frame stack, proprio history) | **High** | Detects “not moving / same view”; standard and usually safe |
| **Progress features** (\(\Delta\) goal dist, EE speed, shield reject flag, time-since-progress) | **High** | Explicit stall detector as input; very practical |
| **Executed actions** \(a^{\text{exec}}_{t-k:t-1}\) (post-shield) | **Medium–high** | Matches what the world actually did; aligns train/deploy if demos also store executed actions under the same shield |
| **Proposed / pre-shield actions** | **Medium** | Useful to learn “I keep proposing illegal → switch”; must log proposals at deploy |
| **Action chunking** (predict \(a_{t:t+H}\), ACT-style) | **High for chatter** | Reduces one-step dithering; related to oracle sequence interventions (V4) |
| **Only last action, no state history** | **Risky** | Classic **causal confusion / copycat**: policy copies \(a_{t-1}\), which under a null-action shield fallback **locks the stall** |

### Critical train/deploy consistency

If deploy executes \(a^{\text{exec}} = \mathrm{Shield}(a^{\text{prop}})\), then history used at train time should be the **same kind of action**:

1. Best: train BC with shield-in-the-loop so demo labels and history are executed actions (ties to Pipeline C).  
2. Weaker: train on expert actions as history but deploy with executed history → distribution shift in the history channel.  
3. Include a binary **shield-rejected** bit in the history; cheap and informative.

### Limits (do not oversell)

- History **disambiguates** stall vs approach; it does **not** create recovery modes absent from \(\mathcal{D}\).  
- Filter-first datasets that dropped near-boundary recoveries still leave history-aware BC with nothing to clone for escape.  
- Longer history ≠ better: too long encourages memorizing teleop quirks; windows of ~2–10 steps or a short chunk horizon are typical starting points.  
- For oscillation against a constraint, **reject flag + progress feature** often beats a long raw action tape.

### Practical recommendation

1. Add **observation/proprio stack + stall/progress features** first (almost always worth it).  
2. Add **executed action history** only if you log shield outputs consistently in demos and rollouts.  
3. Prefer **action chunking** over ever-longer single-step history for tabletop.  
4. Ablate against Markov BC: measure stuck rate and “re-entry into stall after escape,” not just demo MSE.  
5. Still keep a stuck detector + intervention/recovery path — history-aware BC reduces stall frequency; it does not guarantee escape.

---

## 11. Open questions / risks

1. **Projection quality:** In continuous action spaces, “nearest safe action” may be systematically bad (into the constraint surface). When is projection a valid label?  
2. **Stochastic policies / generative IL:** Diffusion / flow policies change how masking and rejection work (sample until safe vs guide). Results may reverse vs deterministic BC.  
3. **Constraint model error:** True safety vs approximate shield — filtering on a wrong model can remove the only actions that were actually safe in reality (or keep unsafe ones).  
4. **Multimodal experts:** Filtering one unsafe mode can erase an entire homotopy class of solutions (left vs right).  
5. **Cost of safe re-collection:** Industrially, collecting new safe demos may dominate both pipelines; academic framing must justify why re-collection is limited.  
6. **Credit assignment for interventions:** Eval-time assists inflate success without producing an autonomous policy — keep these metrics separate.  
7. **Ethical / safety reporting:** Hard shields can hide unsafe nominal policies; deploying with a shield does not mean the BC policy is safe if the shield is removed.  
8. **Theory gap:** Clean PAC-style characterization of when \(F(\mathcal{D})\) preserves sufficiency for safe imitation remains largely open; do not overclaim theory without proofs.  
9. **Sim-to-real:** Shield mismatch (sim collision vs real) may dominate the A vs B effect on hardware.  
10. **History / copycat:** Action-conditioned BC can lock null-action stalls; history design needs ablations (§10).

---

## 12. Pursue or pivot? (recommendation)

### Verdict

**Do not pursue the broad “filter demos vs deploy shield” question as a primary research thesis.**  
**Do pursue a thin engineering pilot** if you need a deployment recipe for SAFE demos — then stop or publish only if a surprising regime split appears.  
**If the goal is a stronger paper, pivot the center of gravity** to one of the sharper adjacent questions below.

### Why not as the main thesis

1. The field already knows design-time vs test-time safety; SAFE-GIL / constraint-aware IL / shielded LfD will dominate related-work comparisons.  
2. A careful A/B/C bake-off is useful lab knowledge but often reads as a **systems ablation**, not a new idea.  
3. Opportunity cost is high: the same nav + tabletop + shield stack can support a tighter claim with better reviewer upside.  
4. The most likely empirical outcome (“C ≥ B ≥ A except when coverage is dense”) is **predictable**, which weakens the paper even if experiments are clean.

### When it *is* still worth a short pursuit

Run a **2–3 week pilot only** if:
- you already have demos + a shield in the stack, and need a go/no-go deployment rule; or
- you specifically want a workshop note / internal design doc, not a flagship paper.

**Kill criteria for the pilot:** if A/B/C rank in the obvious order across constraint severities, and interventions behave like “longer help helps more,” **stop** — do not expand into a full CoRL assault on that dichotomy alone.

### Better bets (ordered)

1. **Best pivot — shield-compatible imitation / path-consistent training:** train so the policy anticipates the shield (projected labels, filter-in-the-loop BC, or diffusion-safe filtering à la path-consistent ideas). Method paper, clearer novelty than A vs B.  
2. **Strong pivot — interventions for shield-induced deadlocks:** treat V3b/V4 as the contribution; baselines = HG-DAgger + Recovery; domains where the shield causes progress failure without safety failure. This is the most distinctive slice of the original idea.  
3. **Data-centric pivot — constraint-aware demo curation:** not blunt drop-illegal, but keep pre-violation context / mine forks / reweight near-boundary states (CoP-Miner-adjacent but for hard deploy constraints on *successful* demos that clip margins).  
4. **Only if you want theory:** characterize when filtering preserves task controllability on the safe set (support / reachability conditions). Higher risk, higher upside at ML venues.

### Practical recommendation for Harshitha / SAFE demonstrations

- **Product / system goal:** implement **B + shield**, upgrade to **C**, add **stuck detector + short oracle sequence (V4) / disagree-until (V3b)**. Do not wait for a paper.  
- **Paper goal:** **do not** lead with filter-vs-constrain. Lead with **(1) or (2)** above; use A vs B as a *motivating ablation* in the experiments section, not the title claim.  
- **Otherwise:** if bandwidth is limited and you care about novelty, **suggest otherwise** — skip the dichotomy study and start from shield-in-the-loop training or deadlock interventions.

---

## 13. Practical bottom line for the project

1. **Implement Pipeline B + hard shield as the production default**, and **Pipeline C (projected labels) as the first training upgrade**.  
2. **Use filter-first only after measuring** that safe demos still cover boundary states; otherwise expect dead-ends.  
3. **Budget oracle sequences (V4) for deep stuckness**; use **V3b** as the default adaptive intervention; keep **V2** as a baseline.  
4. **For a paper:** do **not** sell “filter vs shield” as the idea; sell shield-compatible training or deadlock interventions, with A vs B as motivating evidence.  
5. **Pilot on navigation first** only long enough to hit the kill criteria in §12; then confirm on tabletop or pivot.  
6. **Do not jump to AIRL first** (§9): escalate BC → shield-in-loop BC → masked offline RL → shielded online RL.  
7. **Give BC history** (§10): observation/progress stack first; executed post-shield action history if logged consistently; prefer action chunking over long raw action tapes; ablate copycat risk.

---

## Appendix: Terminology cheat sheet

| Term | Meaning here |
|---|---|
| Filter-first | Remove illegal demo actions/segments, then IL |
| Train-then-constrain | IL on raw demos; shield at deploy |
| Projected training | Relabel illegal demo actions via shield projection, then IL |
| Shield / safety filter | Runtime map from proposed action → safe action / reject |
| Stuck | No task progress under shield, often with high reject rate |
| Oracle intervention | Expert action(s) injected when stuck, optionally aggregated |
