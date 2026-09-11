# RLVR in one file — technical map

Companion to `rlvr_one_file.py`. One section per code block: what goes in, what comes out,
why it exists, and the sentence to say out loud when you draw it.

The file is ~600 lines, stdlib only, and runs with no API key:

```bash
python whiteboard/rlvr_one_file.py          # all five demos end to end
python whiteboard/rlvr_one_file.py trace    # one document, every intermediate number
python whiteboard/rlvr_one_file.py check    # 9 asserts that pin every claim below
```

---

## 0. The 60-second version

> "RLVR means the reward comes from *checking* the answer with code — not from a human rating
> it and not from another model judging it. So the environment is two things: a task with a
> known answer, and a verifier that turns a model's answer into a number between 0 and 1.
> That number then feeds an ordinary policy-gradient update. The interesting engineering is
> almost entirely in the verifier: partial credit so there's a gradient to climb, applicability
> so unannotated fields don't count, and precision-aware scoring so the model can't farm the
> reward. The RL itself is three lines."

Three claims to lead with, in this order:

1. **The verifier is the asset.** Swap the model, keep the verifier. The file's `live` demo
   replaces the fake policy with real Claude and *nothing else changes*.
2. **A reward is a design artifact, not a measurement.** Weights encode what you think matters,
   and a policy will optimize the reward you wrote instead of the task you meant.
3. **Group-relative advantage removes the value network.** Sample K answers to the same task,
   let the group be its own baseline. That's the trick that made RLVR cheap enough to be normal.

---

## 1. The one diagram

Draw this once and refer back to it for the rest of the conversation:

```
   ┌──────────────┐
   │ TASK         │   document + the answer a human already wrote down
   │ (Part 1)     │   ← "known answer" is the whole reason the reward is *verifiable*
   └──────┬───────┘
          │  prompt
          ▼
   ┌──────────────┐
   │ POLICY       │   prompt → text.  An LLM is just a stochastic function here.
   │ (Part 2)     │   temperature = exploration
   └──────┬───────┘
          │  text
          ▼
   ┌──────────────┐
   │ PARSE        │   text → dict.  Failure here = reward 0, not an exception.
   │ (Part 3)     │   forced tool use deletes this stage entirely
   └──────┬───────┘
          │  prediction dict          ground truth dict
          ▼                                  │
   ┌───────────────────────────────────────── ▼ ──┐
   │ VERIFIERS  (Part 4)                          │   3 pure functions:
   │   fields      fuzzy string    → 0.00–1.00    │   (pred, truth) → score, applicable?
   │   money       tolerance       → 0.00–1.00    │
   │   line_items  match + F1      → 0.00–1.00    │   ← no LLM anywhere in this box
   └──────┬───────────────────────────────────────┘
          │  three scores + applicability flags
          ▼
   ┌──────────────┐
   │ COMPOSITE    │   0.30·fields + 0.35·money + 0.35·items
   │ (Part 5)     │   ÷ sum of APPLICABLE weights          → one scalar in [0,1]
   └──────┬───────┘
          │  reward
          ▼
   ┌──────────────┐
   │ ADVANTAGE    │   (r − mean of K siblings) / std        → signed learning signal
   │ (Part 7)     │   no value network, no reward model
   └──────┬───────┘
          │  advantage
          ▼
   ┌──────────────┐
   │ UPDATE       │   θ += lr · advantage · ∇log p(action)
   │ (Part 8)     │   ...and back to the top
   └──────────────┘
```

---

## 2. Block by block

### Part 1 — `Task` and `TASKS`

| | |
|---|---|
| **In** | nothing (hardcoded) |
| **Out** | 3 `Task` objects: `document` (raw text), `truth` (dict), `platform` (metadata) |
| **Say** | "This is the environment. Two boxes: the document, and the answer somebody already wrote down." |

Three synthetic receipts: a direct grocery order, a platform-mediated one (store *and* delivery
app printed — the classic vendor confusion), and a EUR invoice (so "default to USD" is provably
wrong instead of harmlessly wrong).

The schema is 5 scalars + a list — enough to demonstrate string scoring, numeric scoring, and
list scoring. That's every scoring idea there is. The production repo has ~40 fields and teaches
nothing further.

**Where the real cost is:** those `truth` dicts are hand-written. Annotation is the actual
bottleneck in RLVR, not compute. Domains where the answer can be checked *without* annotation —
math (does it equal 42?), code (do the tests pass?) — are where RLVR took off first, and this
env sits in the harder middle: a human had to read each receipt once.

### Part 2 — the policies

| | |
|---|---|
| **In** | a `Task` |
| **Out** | a *string* (the model's raw response, fences and all) |
| **Say** | "A policy maps a document to text. That's all an LLM is in this picture." |

Six hand-written policies stand in for six real failure modes, so the whole file runs offline
**and** you can prove the reward ranks them sensibly:

| policy | behaviour | which verifier catches it |
|---|---|---|
| `careful` | correct transcription | none — scores 1.000 |
| `hurried` | rounds the total, drops the last line item | money + line_items |
| `guesser` | currency always "USD", files the delivery app as the store | fields |
| `lazy` | outputs almost nothing | all three (see reward hacking) |
| `hoarder` | dumps 3 extra plausible rows to never miss one | line_items precision |
| `garbler` | ignores the format and chats | the parse stage |

`lazy` and `hoarder` are not jokes — they are the two shapes reward hacking actually takes, and
keeping them in the eval set is how you notice a reward bug.

### Part 3 — `parse_prediction`

| | |
|---|---|
| **In** | response text |
| **Out** | `dict`, or `None` for malformed |
| **Say** | "Separate stage, because it fails on its own terms. `None` becomes reward 0 — the harness never raises." |

Then the punchline: in production you *delete this stage*. Forced tool use
(`tool_choice={"type": "tool", "name": "submit"}` plus `strict: true`) makes the model's answer
arrive as a dict that already validates against the schema, so "did it produce valid JSON" stops
being a thing you score. See Part 10 in the script for the actual call.

### Part 4 — the three verifiers ← **this is the part that matters**

| | |
|---|---|
| **In** | `(prediction: dict, truth: dict)` |
| **Out** | `Score(name, score ∈ [0,1], applicable: bool, notes)` |
| **Say** | "Pure functions. No LLM in this box. This is what makes the R *verifiable*." |

Three properties, and each one is a design decision an interviewer can push on:

**1. Partial credit.** `string_score` uses character-level similarity; `number_score` is 1.0
inside a one-cent tolerance and then decays with relative error. Binary pass/fail gives a flat
reward landscape with nothing to climb — "GreenLeaf Markt" and "Acme Corp" must not score the
same. *This is the difference between an eval metric and a training signal.*

**2. Applicability.** If the ground truth says nothing about a field, the verifier abstains
(`applicable=False`) rather than scoring 0 or 1. Without this, a blank annotation is
indistinguishable from a perfect answer, and adding annotations looks like a regression.

**3. Scored against the truth, not the prediction.** Every loop iterates over what the *truth*
states. Iterating over what the *model* said is the classic hole: say nothing, be perfect.

The list verifier needs one extra idea — **matching before scoring**. A model may reorder, merge,
or invent rows, so there's no positional alignment to compare. Greedy bipartite matching (best
pair first, remove both, repeat) is 10 lines and near-optimal at receipt scale. Then F1:

```
precision = matched quality / rows the model wrote     ← punishes invention
recall    = matched quality / rows that really exist   ← punishes omission
f1        = harmonic mean
```

Recall alone would pay for guessing. That is exactly what the `hack` demo shows.

### Part 5 — the composite reward

| | |
|---|---|
| **In** | three `Score`s |
| **Out** | `Reward(value, parts, applied, scores)` — one scalar plus full provenance |
| **Say** | "RL needs one number. Collapsing three into one *is* a value judgement, so make it explicit." |

```
reward = Σ (weightᵢ · scoreᵢ) / Σ weightᵢ    over the APPLICABLE verifiers only
```

The renormalization is the subtle bit: drop an inapplicable verifier's weight from the
denominator too, or a receipt with no line items can never score above 0.65.

The weights (`0.30 / 0.35 / 0.35`) are the most arguable thing in the file — which is the point.
When the `eval` demo shows `guesser` (0.953) outranking `hurried` (0.896), that's not a bug:
inventing a currency costs one of three fields inside a 0.30-weight verifier, while dropping a
line item costs half of a 0.35-weight one. If you think hallucination should cost more than
sloppiness, you change the weights or split the verifier. **A composite reward's job is to make
that argument explicit instead of burying it.**

### Part 6 — `Rollout`

| | |
|---|---|
| **In** | `(Task, policy)` |
| **Out** | `Rollout(task, prompt, response, reward)` |
| **Say** | "This record is the unit of RLVR training data. Collect a million of these and you have a dataset." |

### Part 7 — `advantages` (the GRPO trick)

| | |
|---|---|
| **In** | `list[float]` — K rewards for the **same** task |
| **Out** | `list[float]` — signed, mean-0 advantages |
| **Say** | "A reward of 0.71 isn't a signal. Good compared to what? Let the siblings answer." |

```
advantageᵢ = (rewardᵢ − mean(rewards)) / std(rewards)
```

Classic RL answers "compared to what" with a learned value network. GRPO answers it with
sampling: draw K answers to the same task, and the group is its own baseline. No value network,
no reward model, no human preference labels — just K samples and a verifier. That's the cost
collapse that made this approach standard.

Two consequences worth volunteering before you're asked:

- **A unanimous group produces zero gradient.** std = 0 ⇒ every advantage is 0. All-right and
  all-wrong groups are wasted compute, which is why task difficulty and curriculum are
  first-class concerns in a real RLVR pipeline.
- **The baseline is relative, so group composition is part of the algorithm.** In the `group`
  demo, `hoarder` gets a *positive* advantage purely because one terrible sibling drags the mean
  down. Everything above the mean gets reinforced, mediocre or not.

### Part 8 — the update (REINFORCE, four visible parameters)

| | |
|---|---|
| **In** | `steps`, `group` (K), `lr` |
| **Out** | converged probabilities per habit + history |
| **Say** | "You can't watch 70B weights move on a whiteboard. So here's a policy with four parameters that learns from exactly the same signal." |

The policy is four independent yes/no habits, each with a logit θ, so `p = sigmoid(θ)`. Sampling
a habit vector is the analogue of sampling tokens. The habits are real extraction failure modes:

| habit | should converge to |
|---|---|
| `read_currency` — transcribe the printed code | 1.0 |
| `platform_as_vendor` — file the delivery app as the store | 0.0 |
| `pad_items` — add a plausible unlisted row | 0.0 |
| `round_total` — round to whole units | 0.0 |

For a Bernoulli policy, ∇<sub>θ</sub> log p(action) is exactly `(action − p)`, so the whole
update is:

```python
theta[h] += lr * mean over group of [ advantage * (action - p) ]
```

That line is the entire "RL" in RLVR. Observed output after 40 steps × 8 rollouts:

```
read_currency        p=0.986   (target 1)
platform_as_vendor   p=0.067   (target 0)
pad_items            p=0.017   (target 0)
round_total          p=0.025   (target 0)
```

**The sentence that lands:** nothing in that loop was told what the right answers were. The
verifier knew; the advantage carried it into the parameters.

One honest disclosure to make unprompted: `render_with_habits` assumes the *reading* of the
document is correct so the demo isolates the four habits. A real policy's mistakes come from the
same place — sampled tokens — and reach the verifier through the same channel.

### Part 9 — reward hacking, and the two lines that stop it

| | |
|---|---|
| **In** | the same policies, scored by a deliberately buggy `naive_reward` |
| **Out** | the exploit, in numbers |
| **Say** | "A policy optimizes the reward you wrote, not the task you meant." |

Two plausible-sounding verifier bugs, each with a policy that eats it:

| bug | how it sounds when you write it | who exploits it | naive → designed |
|---|---|---|---|
| score the fields the model *predicted* | "don't punish it for what it didn't attempt" | `lazy` | **1.000 → 0.100** |
| score line items by recall alone | "reward finding everything" | `hoarder` | **1.000 → 0.883** |

Both fixes are one line each: iterate over the truth's fields, and use F1 instead of recall.

The real lesson is the third row of that table: on `careful`, both rewards return 1.000. **A
reward bug is invisible until something optimizes against it.** Hence adversarial baselines in
the eval set — a lazy policy and a hoarding policy are your canaries, and they cost nothing to
keep around.

### Part 10 — the same verifiers, a real model

Nothing above the parse stage cares where the text came from:

```python
client.messages.create(
    model="claude-opus-5",
    tools=[SUBMIT_TOOL],                                   # strict: true
    tool_choice={"type": "tool", "name": "submit"},         # forced → answer IS a tool call
    messages=[{"role": "user", "content": prompt_for(task)}],
)
```

Forced tool use means schema-valid structured output is a property of the API call rather than
something you parse and hope for. `python whiteboard/rlvr_one_file.py live` runs it (needs
`anthropic` and credentials) and prints the same reward breakdown.

---

## 3. The numbers to have in your head

From `trace` — one rollout of `hurried` on the platform-mediated receipt:

```
fields      1.000    vendor ✓  order_id ✓  currency ✓
money       0.995    tax 1.32 ✓;  total 22 vs 21.80 → 1 − 0.20/21.80 = 0.99
line_items  0.667    1 of 2 matched → precision 1.00, recall 0.50 → F1 0.67

reward = (0.30·1.000 + 0.35·0.995 + 0.35·0.667) / 1.00 = 0.882
```

From `eval` — the reward ranks behaviour the way a human would:

```
careful 1.000 > guesser 0.953 > hurried 0.896 > hoarder 0.861 > lazy 0.100 > garbler 0.000
```

Read the *columns*, not the mean: that's the difference between a leaderboard number and a
diagnosis. `garbler` at 0.000 everywhere is the format failure; `lazy` at 0.100 is what
"scored against the truth" buys you.

---

## 4. Questions you will get

**"Isn't this just an eval harness with extra steps?"**
An eval is the same code with the last two blocks deleted. The differences that matter for
training: partial credit (a flat reward has no gradient), applicability (so incremental
annotation isn't a regression), and per-component breakdowns (so you can reweight what the model
is actually rewarded for). An eval can be pass/fail; a reward can't.

**"Why not LLM-as-judge?"**
Then the reward is a model output — noisy, expensive per rollout, drifting between versions, and
game-able by the policy it's grading. Deterministic verifiers are reproducible and roughly free,
which matters when you need one reward per rollout for millions of rollouts. The cost is
coverage: you can only verify what you annotated, which is why `DetailVerifier` in the full repo
reports unannotated extras instead of punishing them.

**"How do you know your reward isn't hacked?"**
You don't, from the benign policies — they agree with a buggy reward. You keep adversarial
baselines (`lazy`, `hoarder`) in the eval set and watch for the gap. Volunteer the `hack` demo
before they ask; it's the most senior-sounding thing in the file.

**"Where do the weights come from?"**
Judgement, then argument. They encode which errors you consider expensive. The `guesser` >
`hurried` ranking is the live example: I can defend the current weights or change them, but
either way the tradeoff is one dict, in one place, with a printed breakdown showing its effect.

**"Why GRPO and not PPO?"**
PPO needs a value network — another model to train, hold in memory, and debug. When you can
cheaply sample K answers to the same prompt, the group mean is a good enough baseline, and the
std normalizes the scale. Fewer moving parts for the same variance reduction.

**"What breaks at scale?"**
Annotation coverage first: verifiers can only score what's annotated, so the reward silently
stops covering new fields — the full repo has a drift-guard test that fails when a schema field
no verifier scores is added. Then group composition: as the policy improves, more groups come
back unanimous and stop producing gradient, so you need difficulty curation. Then the usual
distribution shift — a verifier tuned on receipts that print quantities behaves differently on
ones that don't.

---

## 5. What the full repo adds (and why none of it is in here)

`../src/info_extract/` is the same seven blocks, hardened:

| here | there | why the difference exists |
|---|---|---|
| dict predictions | Pydantic `InvoiceExtraction` — one model that is *simultaneously* the tool schema, the response validator, the ground-truth type, and both verifier inputs | one source of truth for the shape |
| 6 fields | ~40, incl. per-item identifiers, taxonomy, split tenders, instalments | real receipts |
| 3 verifiers | 4, with a `DetailVerifier` that reports unannotated extras rather than punishing them | report-don't-punish |
| synthetic text | `.eml`/`.pdf` parsers + two-tier PII redaction, plus a schema with nowhere for PII to land | real documents contain real people |
| 9 asserts | 221 tests, incl. a drift guard that fails when a schema field no verifier scores is added | the silent-drop failure mode |

**Say this if the size comes up:** "The one-file version is the version I'd defend on a
whiteboard. The repo is the same seven blocks with the parts that only matter once real
documents and real annotations are involved — PII, schema drift, parser edge cases."
