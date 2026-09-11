#!/usr/bin/env python3
"""
rlvr_one_file.py — a complete RLVR environment in one file, built to be drawn on a whiteboard.

RLVR = Reinforcement Learning with Verifiable Rewards. The whole idea in one line:

    the reward comes from *checking* the answer with code, not from a human rating it
    and not from another model judging it.

The loop, which is the only picture you need:

    task ──▶ policy ──▶ response ──▶ parse ──▶ verifiers ──▶ reward ──▶ advantage ──▶ update
     │                  (text)      (dict)    (code, no LLM)  [0,1]    (vs. siblings)   │
     └──────────────────────────────────────────────────────────────────────────────────┘

Every stage below is one small section of this file. Nothing is imported from the big
`src/info_extract` package on purpose — this file is the explanation of that package.

Run it:

    python whiteboard/rlvr_one_file.py              # all five demos, no API key needed
    python whiteboard/rlvr_one_file.py trace        # one document, every number shown
    python whiteboard/rlvr_one_file.py eval         # 6 policies x 3 tasks reward matrix
    python whiteboard/rlvr_one_file.py group        # GRPO-style group-relative advantage
    python whiteboard/rlvr_one_file.py train        # REINFORCE: rewards actually move a policy
    python whiteboard/rlvr_one_file.py hack         # reward hacking, and the two guards
    python whiteboard/rlvr_one_file.py check        # 9 asserts that pin the claims above
    python whiteboard/rlvr_one_file.py live         # same verifiers, real Claude (needs a key)

Stdlib only (the `live` demo needs `anthropic`). No torch, no numpy, no framework.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# =====================================================================================
# PART 1 — THE TASK
# -------------------------------------------------------------------------------------
# An RL "environment" needs a task distribution. Here a task is: a document to read, and
# the answer a human already wrote down. The answer is what makes the reward *verifiable*.
#
# Whiteboard: two boxes. [ DOCUMENT ]  and  [ GROUND TRUTH ]. That's the environment.
# =====================================================================================


@dataclass
class Task:
    """One (input, known-answer) pair. `platform` is metadata used by the training demo."""

    name: str
    document: str
    truth: dict
    platform: str | None = None


TASKS: list[Task] = [
    Task(
        name="grocery-direct",
        document="""
        GreenLeaf Market
        Order #GL-88213
        2026-03-04

        PRODUCE
          Bananas            1.24 lb        1.86
          Roma Tomatoes      2              4.98
        PANTRY
          Oat Milk 64oz      1              5.49

        Subtotal                           12.33
        Tax                                 0.86
        Total                              13.19
        Paid with VISA
        """,
        truth={
            "vendor": "GreenLeaf Market",
            "order_id": "GL-88213",
            "currency": "USD",
            "tax": 0.86,
            "total": 13.19,
            "line_items": [
                {"name": "Bananas", "qty": 1.24, "price": 1.86},
                {"name": "Roma Tomatoes", "qty": 2.0, "price": 4.98},
                {"name": "Oat Milk 64oz", "qty": 1.0, "price": 5.49},
            ],
        },
    ),
    Task(
        name="platform-mediated",
        document="""
        QuickCart Delivery
        Your order from Corner Pharmacy
        Order QC-4471 - Mar 6, 2026

        Ibuprofen 200ct     1     9.99
        Cough Drops         2     6.50

        Subtotal                16.49
        Delivery fee             3.99
        Tax                      1.32
        Total                   21.80
        """,
        truth={
            "vendor": "Corner Pharmacy",  # the store the goods came from
            "order_id": "QC-4471",
            "currency": "USD",
            "tax": 1.32,
            "total": 21.80,
            "line_items": [
                {"name": "Ibuprofen 200ct", "qty": 1.0, "price": 9.99},
                {"name": "Cough Drops", "qty": 2.0, "price": 6.50},
            ],
        },
        platform="QuickCart Delivery",  # ...not the app that delivered them
    ),
    Task(
        name="euro-invoice",
        document="""
        Papeterie Lumiere
        Facture 2026-0219
        Montant en EUR

        Carnet A5           3     14.70
        Stylo gel           1      2.30

        Sous-total                17.00
        TVA (20%)                  3.40
        Total                     20.40
        """,
        truth={
            "vendor": "Papeterie Lumiere",
            "order_id": "2026-0219",
            "currency": "EUR",  # printed, so it must be transcribed — never defaulted to USD
            "tax": 3.40,
            "total": 20.40,
            "line_items": [
                {"name": "Carnet A5", "qty": 3.0, "price": 14.70},
                {"name": "Stylo gel", "qty": 1.0, "price": 2.30},
            ],
        },
    ),
]

# The schema is deliberately tiny: 5 scalars + a list. Six fields is enough to demonstrate
# every scoring idea (string, number, list); the production repo has ~40 and learns nothing new.
SCHEMA_HINT = """{
  "vendor": str|null, "order_id": str|null, "currency": str|null,
  "tax": float|null, "total": float|null,
  "line_items": [{"name": str, "qty": float|null, "price": float|null}]
}"""


# =====================================================================================
# PART 2 — THE POLICY
# -------------------------------------------------------------------------------------
# A policy maps a document to *text*. That's all an LLM is in this picture: a stochastic
# function from prompt to string. Below, six hand-written policies stand in for six
# behaviours a real model exhibits, so the whole file runs offline and you can *show*
# that the reward ranks them correctly. `live` swaps in the real thing.
#
# Whiteboard: one box, [ POLICY ], with a dial on it labelled "temperature".
# =====================================================================================


def prompt_for(task: Task) -> str:
    """The prompt. Not the interesting part of RLVR, but it is part of the environment."""
    return (
        "Transcribe this receipt into JSON. Only what is printed — never infer a value.\n"
        f"Schema:\n{SCHEMA_HINT}\n\nDocument:\n{task.document}"
    )


def _as_json(obj: dict) -> str:
    """Policies return text, like a model does — fences included, because parsing is a stage."""
    return "```json\n" + json.dumps(obj, indent=2) + "\n```"


def policy_careful(task: Task) -> str:
    """Reads the document correctly."""
    return _as_json(task.truth)


def policy_hurried(task: Task) -> str:
    """Right vendor, but rounds the total and skips the last line item."""
    out = json.loads(json.dumps(task.truth))
    out["total"] = round(out["total"])
    out["line_items"] = out["line_items"][:-1]
    return _as_json(out)


def policy_guesser(task: Task) -> str:
    """Fills blanks with plausible defaults: currency is always USD, platform filed as store."""
    out = json.loads(json.dumps(task.truth))
    out["currency"] = "USD"
    if task.platform:
        out["vendor"] = task.platform
    return _as_json(out)


def policy_lazy(task: Task) -> str:
    """Says almost nothing. Safe under a badly designed reward — see the `hack` demo."""
    return _as_json({"vendor": task.truth["vendor"], "line_items": []})


def policy_hoarder(task: Task) -> str:
    """Dumps every plausible line item to make sure it never misses one. Farms recall."""
    out = json.loads(json.dumps(task.truth))
    out["line_items"] += [
        {"name": "Reusable Bag", "qty": 1.0, "price": 0.10},
        {"name": "Bottle Deposit", "qty": 1.0, "price": 0.05},
        {"name": "Loyalty Discount", "qty": 1.0, "price": -1.00},
    ]
    return _as_json(out)


def policy_garbler(task: Task) -> str:
    """Ignores the format and chats. Must score 0 without crashing the harness."""
    return "Sure! This looks like a receipt from a shop. The total seems to be about twenty."


POLICIES = {
    "careful": policy_careful,
    "hurried": policy_hurried,
    "guesser": policy_guesser,
    "lazy": policy_lazy,
    "hoarder": policy_hoarder,
    "garbler": policy_garbler,
}


# =====================================================================================
# PART 3 — PARSE
# -------------------------------------------------------------------------------------
# Text -> dict. A separate stage because it fails on its own terms: an unparseable answer
# is reward 0, not an exception. In production you avoid this stage entirely by forcing
# tool use (see `live`), which makes schema-valid JSON a property of the API call.
# =====================================================================================


def parse_prediction(response: str) -> dict | None:
    """Pull the first JSON object out of a model response. None means 'malformed'."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.S)
    blob = fenced.group(1) if fenced else None
    if blob is None:
        start = response.find("{")
        blob = response[start:] if start >= 0 else None
    if blob is None:
        return None
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# =====================================================================================
# PART 4 — THE VERIFIERS
# -------------------------------------------------------------------------------------
# The heart of RLVR. Each verifier is a pure function
#
#     (prediction, truth) -> score in [0,1], plus a note, plus "was this applicable?"
#
# Three properties are what make these usable as a *training* signal rather than a grade:
#
#   1. PARTIAL CREDIT. "GreenLeaf Markt" is closer than "Acme Corp". Binary pass/fail gives
#      a flat reward landscape with nothing to climb.
#   2. APPLICABILITY. If the truth says nothing about a field, the verifier abstains instead
#      of scoring 0 or 1. Otherwise a blank annotation is indistinguishable from a perfect
#      answer, and adding annotations looks like a regression.
#   3. SCORED AGAINST THE TRUTH, NOT THE PREDICTION. Every loop below iterates over what the
#      truth states. Iterating over what the model said is the classic reward-hacking hole
#      (say nothing, be perfect) — demonstrated in the `hack` demo.
#
# Whiteboard: three small boxes feeding one adder.
# =====================================================================================


@dataclass
class Score:
    """One verifier's output."""

    name: str
    score: float
    applicable: bool = True
    notes: list[str] = field(default_factory=list)


def string_score(pred: str | None, truth: str) -> float:
    """Partial credit for strings: character-level similarity in [0,1]."""
    if not pred:
        return 0.0
    return SequenceMatcher(None, pred.lower().strip(), truth.lower().strip()).ratio()


def number_score(pred: float | None, truth: float, tol: float = 0.01) -> float:
    """1.0 inside tolerance, then decaying with relative error. 'Off by a cent' != 'off by 10x'."""
    if pred is None or not isinstance(pred, (int, float)):
        return 0.0
    diff = abs(float(pred) - truth)
    if diff <= tol:
        return 1.0
    if truth == 0:
        return 0.0
    return max(0.0, 1.0 - diff / abs(truth))


def verify_fields(pred: dict, truth: dict) -> Score:
    """Verifier 1 — the scalar text fields. Fuzzy match, averaged over stated fields."""
    parts, notes = [], []
    for key in ("vendor", "order_id", "currency"):
        if truth.get(key) in (None, ""):
            continue  # not stated by the truth -> not scored (property 2)
        s = string_score(pred.get(key), truth[key])
        parts.append(s)
        notes.append(f"{key}: {pred.get(key)!r} vs {truth[key]!r} = {s:.2f}")
    if not parts:
        return Score("fields", 0.0, applicable=False, notes=["nothing annotated"])
    return Score("fields", sum(parts) / len(parts), notes=notes)


def verify_money(pred: dict, truth: dict) -> Score:
    """Verifier 2 — the monetary fields. Tolerance-based, because floats and cents."""
    parts, notes = [], []
    for key in ("tax", "total"):
        if truth.get(key) is None:
            continue
        s = number_score(pred.get(key), truth[key])
        parts.append(s)
        notes.append(f"{key}: {pred.get(key)!r} vs {truth[key]!r} = {s:.2f}")
    if not parts:
        return Score("money", 0.0, applicable=False, notes=["nothing annotated"])
    return Score("money", sum(parts) / len(parts), notes=notes)


def item_similarity(pred_item: dict, truth_item: dict) -> float:
    """How much one predicted line item looks like one true line item. Half name, half price."""
    name = string_score(pred_item.get("name"), truth_item.get("name", ""))
    if truth_item.get("price") is None:
        return name
    return 0.5 * name + 0.5 * number_score(pred_item.get("price"), truth_item["price"])


def verify_line_items(pred: dict, truth: dict) -> Score:
    """Verifier 3 — the list. Greedy one-to-one matching, then F1.

    A list needs matching before scoring: the model may reorder, merge, or invent rows, so
    there is no positional alignment to compare. Greedy (best pair first, then remove both)
    is 10 lines and within a hair of optimal at receipt scale; the real repo does the same.

    F1 and not recall, because recall alone pays for guessing (see the `hack` demo):
        precision = matched quality / how many rows the model wrote   (punishes invention)
        recall    = matched quality / how many rows really exist      (punishes omission)
    """
    true_items = truth.get("line_items") or []
    pred_items = [i for i in (pred.get("line_items") or []) if isinstance(i, dict)]
    if not true_items:
        return Score("line_items", 0.0, applicable=False, notes=["no annotated items"])
    if not pred_items:
        return Score("line_items", 0.0, notes=[f"predicted 0 of {len(true_items)} items"])

    pairs = sorted(
        (
            (item_similarity(p, t), i, j)
            for i, p in enumerate(pred_items)
            for j, t in enumerate(true_items)
        ),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_true: set[int] = set()
    matched = 0.0
    notes = []
    for sim, i, j in pairs:
        if sim < 0.3 or i in used_pred or j in used_true:
            continue
        used_pred.add(i)
        used_true.add(j)
        matched += sim
        notes.append(f"{pred_items[i].get('name')!r} ~ {true_items[j].get('name')!r} = {sim:.2f}")

    precision = matched / len(pred_items)
    recall = matched / len(true_items)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    notes.append(f"precision={precision:.2f} recall={recall:.2f} f1={f1:.2f}")
    return Score("line_items", f1, notes=notes)


VERIFIERS = (verify_fields, verify_money, verify_line_items)


# =====================================================================================
# PART 5 — THE COMPOSITE REWARD
# -------------------------------------------------------------------------------------
# RL needs one scalar. Three verifiers must collapse to a single number, and the collapse
# encodes a value judgement: what does "good" mean here?
#
#     reward = sum(weight_i * score_i) / sum(weight_i)   over the APPLICABLE verifiers only
#
# The renormalization is the subtle part. Drop an inapplicable verifier's weight from the
# denominator too, or a document with no line items can never score above 0.65.
#
# Whiteboard: three arrows into a circle labelled "0.30 / 0.35 / 0.35", one arrow out.
# =====================================================================================

WEIGHTS = {"fields": 0.30, "money": 0.35, "line_items": 0.35}


@dataclass
class Reward:
    """The scalar the RL algorithm consumes, plus everything needed to debug it."""

    value: float
    parts: dict[str, float] = field(default_factory=dict)
    applied: list[str] = field(default_factory=list)
    scores: list[Score] = field(default_factory=list)
    note: str = ""


def compute_reward(response: str, truth: dict) -> Reward:
    """response text -> scalar reward. This function *is* the environment's step()."""
    pred = parse_prediction(response)
    if pred is None:
        # A malformed answer is a bad answer, not a crash. Same signal shape, value 0.
        return Reward(0.0, note="unparseable response")

    scores = [v(pred, truth) for v in VERIFIERS]
    applied = [s.name for s in scores if s.applicable]
    total_weight = sum(WEIGHTS[n] for n in applied)
    value = (
        sum(s.score * WEIGHTS[s.name] for s in scores if s.applicable) / total_weight
        if total_weight
        else 0.0
    )
    return Reward(
        value=value,
        parts={s.name: s.score for s in scores},
        applied=applied,
        scores=scores,
    )


# =====================================================================================
# PART 6 — ROLLOUT
# -------------------------------------------------------------------------------------
# One (task, policy) pass through everything above. This record — prompt, response, reward
# — is the unit of RLVR training data. Collect a million of these and you have a dataset.
# =====================================================================================


@dataclass
class Rollout:
    task: str
    prompt: str
    response: str
    reward: Reward


def rollout(task: Task, policy) -> Rollout:
    prompt = prompt_for(task)
    response = policy(task)
    return Rollout(task.name, prompt, response, compute_reward(response, task.truth))


# =====================================================================================
# PART 7 — FROM REWARD TO LEARNING SIGNAL: GROUP-RELATIVE ADVANTAGE
# -------------------------------------------------------------------------------------
# A raw reward of 0.71 is not a training signal — is that good? Compared to what? Classic
# RL answers with a learned value network ("predict the average reward from here"). GRPO
# (what DeepSeek-R1 and most RLVR pipelines use) answers with sampling instead: draw K
# answers to the SAME task, and let the group be its own baseline.
#
#     advantage_i = (reward_i - mean(rewards)) / std(rewards)
#
# Above average -> positive -> make that answer more likely. Below -> negative -> less.
# No value network, no reward model, no human labels. Just K samples and a verifier.
#
# One consequence worth knowing: if all K rewards are identical, std is 0, every advantage
# is 0, and the group produces NO gradient. Groups that are all-right or all-wrong are
# wasted compute — which is why curriculum and task difficulty matter so much in practice.
# =====================================================================================


def advantages(rewards: list[float]) -> list[float]:
    """Group-relative advantage. Returns all zeros when the group is unanimous."""
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0] * n  # unanimous group -> no signal at all
    return [(r - mean) / std for r in rewards]


# =====================================================================================
# PART 8 — THE UPDATE: REINFORCE ON A POLICY YOU CAN SEE
# -------------------------------------------------------------------------------------
# The last piece of a real RL env is the part that changes the policy. You cannot watch
# 70B weights move on a whiteboard, so here the policy has FOUR parameters. It is still a
# genuine stochastic policy trained by genuine policy gradient — the mechanics below are
# exactly what a GRPO trainer does, with `theta` standing in for the weights.
#
# The policy: four independent yes/no habits, each with a logit theta -> p = sigmoid(theta).
# Sampling a habit vector is the analogue of sampling tokens. The habits are the ones a
# real extractor gets wrong:
#
#   read_currency      transcribe the printed currency code   (should go UP -> 1.0)
#   platform_as_vendor file the delivery app as the store     (should go DOWN -> 0.0)
#   pad_items          add a plausible unlisted line item     (should go DOWN -> 0.0)
#   round_total        round the total to whole units         (should go DOWN -> 0.0)
#
# REINFORCE with the group baseline. For a Bernoulli policy the gradient of
# log p(action) w.r.t. theta is exactly (action - p), so:
#
#     theta += lr * mean over group of [ advantage * (action - p) ]
#
# That one line is the entire "RL" in RLVR. Everything else is the verifier.
# =====================================================================================

HABITS = ("read_currency", "platform_as_vendor", "pad_items", "round_total")
GOOD_HABIT = {"read_currency": True, "platform_as_vendor": False, "pad_items": False, "round_total": False}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def render_with_habits(task: Task, on: dict[str, bool]) -> str:
    """Turn a habit vector into an answer.

    Shortcut, stated plainly: the *reading* of the document is assumed correct, so the demo
    isolates the four habits. A real policy's mistakes come from the same place — sampled
    tokens — and reach the verifier through the same channel.
    """
    out = json.loads(json.dumps(task.truth))
    if not on["read_currency"]:
        out["currency"] = None
    if on["platform_as_vendor"] and task.platform:
        out["vendor"] = task.platform
    if on["pad_items"]:
        out["line_items"].append({"name": "Reusable Bag", "qty": 1.0, "price": 0.10})
    if on["round_total"]:
        out["total"] = float(round(out["total"]))
    return _as_json(out)


def train(steps: int = 40, group: int = 8, lr: float = 0.8, seed: int = 0, verbose: bool = True):
    """The full loop: sample a group, verify, normalize, nudge. Returns final probabilities."""
    rng = random.Random(seed)
    theta = dict.fromkeys(HABITS, 0.0)  # p = 0.5 for every habit: the policy knows nothing
    history = []

    for step in range(1, steps + 1):
        task = rng.choice(TASKS)  # one task per step; the group shares it (that's the baseline)
        probs = {h: sigmoid(theta[h]) for h in HABITS}

        # --- collect a group of rollouts on the SAME task -------------------------------
        actions = [{h: rng.random() < probs[h] for h in HABITS} for _ in range(group)]
        rewards = [compute_reward(render_with_habits(task, a), task.truth).value for a in actions]

        # --- turn rewards into advantages ----------------------------------------------
        adv = advantages(rewards)

        # --- policy gradient step ------------------------------------------------------
        for h in HABITS:
            grad = sum(a * (1.0 if act[h] else 0.0) - a * probs[h] for a, act in zip(adv, actions))
            theta[h] += lr * grad / group

        history.append((step, task.name, sum(rewards) / group, dict(probs)))
        if verbose and (step == 1 or step % 5 == 0):
            unanimous = " (unanimous group: zero gradient)" if all(a == 0.0 for a in adv) else ""
            bars = "  ".join(f"{h}={sigmoid(theta[h]):.2f}" for h in HABITS)
            print(f"  step {step:>3}  task={task.name:<18} mean_reward={sum(rewards)/group:.3f}"
                  f"  {bars}{unanimous}")

    return {h: sigmoid(theta[h]) for h in HABITS}, history


# =====================================================================================
# PART 9 — REWARD HACKING, AND THE TWO LINES THAT STOP IT
# -------------------------------------------------------------------------------------
# A policy optimizes the reward you wrote, not the task you meant. Below is the same
# environment with two plausible-looking verifier bugs, each of which a policy exploits:
#
#   BUG 1: score the fields the model *predicted* instead of the fields the truth states.
#          Sounds like "don't punish it for what it didn't try". A policy that outputs
#          almost nothing then scores 1.00.
#   BUG 2: score line items by recall alone. Sounds like "reward finding everything".
#          A policy that dumps every plausible row then scores 1.00.
#
# Both bugs are invisible on a benign policy and obvious the moment something optimizes
# against them. That asymmetry is the whole reason to keep a lazy and a hoarding baseline
# in your eval set: they are the canaries for reward design.
# =====================================================================================


def naive_reward(response: str, truth: dict) -> float:
    """The buggy reward. Kept deliberately short so the two bugs are visible at a glance."""
    pred = parse_prediction(response)
    if pred is None:
        return 0.0
    parts = []
    for key in ("vendor", "order_id", "currency"):
        if pred.get(key):  # BUG 1: iterate over the PREDICTION, so silence is never wrong
            parts.append(string_score(pred[key], truth.get(key) or ""))
    for key in ("tax", "total"):
        if pred.get(key) is not None:
            parts.append(number_score(pred[key], truth.get(key) or 0.0))
    true_items = truth.get("line_items") or []
    pred_items = [i for i in (pred.get("line_items") or []) if isinstance(i, dict)]
    if true_items and pred_items:
        best = sum(max(item_similarity(p, t) for p in pred_items) for t in true_items)
        parts.append(best / len(true_items))  # BUG 2: recall only, invention is free
    return sum(parts) / len(parts) if parts else 1.0  # and an empty answer is "perfect"


# =====================================================================================
# PART 10 — THE SAME VERIFIERS AGAINST A REAL MODEL
# -------------------------------------------------------------------------------------
# Nothing above the parse stage cares where the text came from. Swap the fake policy for
# Claude and the harness is unchanged — which is the point: the verifier is the asset.
#
# Forced tool use replaces the parse stage: the API is told it must call `submit`, so the
# response arrives as a dict that already validates against the schema. `strict: true`
# makes that a guarantee rather than a strong tendency.
# =====================================================================================

MODEL = "claude-opus-5"

SUBMIT_TOOL = {
    "name": "submit",
    "description": "Submit the transcribed receipt.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["vendor", "order_id", "currency", "tax", "total", "line_items"],
        "properties": {
            "vendor": {"type": ["string", "null"]},
            "order_id": {"type": ["string", "null"]},
            "currency": {"type": ["string", "null"]},
            "tax": {"type": ["number", "null"]},
            "total": {"type": ["number", "null"]},
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "qty", "price"],
                    "properties": {
                        "name": {"type": "string"},
                        "qty": {"type": ["number", "null"]},
                        "price": {"type": ["number", "null"]},
                    },
                },
            },
        },
    },
}


def claude_policy(task: Task) -> str:
    """A real policy. Returns JSON text so the rest of the file is untouched."""
    import anthropic  # imported lazily: every other demo runs without the SDK

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        tools=[SUBMIT_TOOL],
        tool_choice={"type": "tool", "name": "submit"},  # forced: the answer must be a tool call
        messages=[{"role": "user", "content": prompt_for(task)}],
    )
    for block in message.content:
        if block.type == "tool_use":
            return json.dumps(block.input)
    return "".join(b.text for b in message.content if b.type == "text")


# =====================================================================================
# PART 11 — THE DEMOS
# =====================================================================================


def rule(title: str) -> None:
    print(f"\n{'=' * 84}\n{title}\n{'=' * 84}")


def demo_trace() -> None:
    """One document, one policy, every intermediate value. The whiteboard walkthrough."""
    rule("TRACE — one rollout, every number")
    task, policy_name = TASKS[1], "hurried"
    roll = rollout(task, POLICIES[policy_name])

    print(f"\n[1] TASK  {task.name}   policy={policy_name}")
    print("    document (excerpt):", " / ".join(task.document.split()[:9]), "...")
    print(f"\n[2] RESPONSE (text, {len(roll.response)} chars)")
    print("   ", roll.response.replace("\n", " ")[:150], "...")
    print("\n[3] PARSE  text -> dict")
    pred = parse_prediction(roll.response)
    print(f"    keys={sorted(pred)}  line_items={len(pred['line_items'])}")

    print("\n[4] VERIFY  each verifier is a pure function of (prediction, truth)")
    for s in roll.reward.scores:
        flag = "" if s.applicable else "  [INAPPLICABLE — dropped, weight renormalized]"
        print(f"    {s.name:<11} score={s.score:.3f}{flag}")
        for note in s.notes:
            print(f"        {note}")

    print("\n[5] COMPOSE  weighted mean over applicable verifiers")
    terms = " + ".join(f"{WEIGHTS[n]:.2f}*{roll.reward.parts[n]:.3f}" for n in roll.reward.applied)
    denom = sum(WEIGHTS[n] for n in roll.reward.applied)
    print(f"    reward = ({terms}) / {denom:.2f} = {roll.reward.value:.3f}")
    print("\n    That 0-to-1 number is the only thing the RL algorithm ever sees.")


def demo_eval() -> None:
    """The eval-harness view: does the reward rank behaviours the way a human would?"""
    rule("EVAL — 6 policies x 3 tasks. This is what 'evaluating an LLM' means here.")
    names = list(POLICIES)
    header = "  ".join(f"{n:>9}" for n in names)
    print(f"\n  {'task':<20}{header}")
    means = dict.fromkeys(names, 0.0)
    for task in TASKS:
        row = []
        for n in names:
            r = rollout(task, POLICIES[n]).reward.value
            means[n] += r / len(TASKS)
            row.append(f"{r:>9.3f}")
        print(f"  {task.name:<20}{'  '.join(row)}")
    print(f"  {'-' * (20 + len(header))}")
    print(f"  {'MEAN':<20}{'  '.join(f'{means[n]:>9.3f}' for n in names)}")

    ranked = sorted(means, key=means.get, reverse=True)
    print(f"\n  ranking: {' > '.join(ranked)}")
    print("  Read the failure modes off the columns, not just the mean:")
    print("    hurried  loses money + line_items   (rounding, dropped row)")
    print("    guesser  loses fields only, and only on the two tasks it can be wrong about")
    print("    lazy     scores near 0 because verifiers iterate over the TRUTH")
    print("    hoarder  keeps recall 1.0 but F1 falls — invention is not free")
    print("    garbler  0.000 everywhere: unparseable is a score, not an exception")
    print("\n  Own the uncomfortable row: `guesser` outranks `hurried`. Not a bug — a weights")
    print("  statement. Inventing a currency costs one of three fields inside a 0.30 verifier;")
    print("  dropping a line item costs half of a 0.35 one. If you believe hallucination")
    print("  should cost more than sloppiness, you change WEIGHTS or split the field verifier.")
    print("  The composite's job is to make that argument explicit instead of implicit.")


def demo_group() -> None:
    """Where the learning signal comes from when there is no value network."""
    rule("GROUP — reward -> advantage. The group is its own baseline (GRPO).")
    task = TASKS[0]
    mixed = ["careful", "careful", "hurried", "guesser", "hoarder", "lazy"]
    rewards = [rollout(task, POLICIES[p]).reward.value for p in mixed]
    adv = advantages(rewards)

    print(f"\n  task={task.name}, K={len(mixed)} sampled answers\n")
    print(f"  {'sample':<10}{'reward':>9}{'advantage':>12}   direction")
    for p, r, a in zip(mixed, rewards, adv):
        arrow = "make MORE likely" if a > 0 else ("make LESS likely" if a < 0 else "no change")
        print(f"  {p:<10}{r:>9.3f}{a:>12.2f}   {arrow}")
    mean = sum(rewards) / len(rewards)
    print(f"\n  mean={mean:.3f}  std={math.sqrt(sum((r-mean)**2 for r in rewards)/len(rewards)):.3f}")
    print("\n  Note `hoarder` gets a POSITIVE advantage here: one terrible sibling (`lazy`)")
    print("  drags the mean down, and everything above the mean is reinforced. The baseline")
    print("  is relative, so group composition is part of the algorithm, not a detail.")

    same = [0.71] * 6
    print(f"\n  Degenerate case — all K rewards equal: advantages={advantages(same)}")
    print("  Unanimous group => zero gradient => wasted compute. Task difficulty is a")
    print("  first-class concern in RLVR for exactly this reason.")


def demo_train() -> None:
    """The loop closed: verified rewards actually move a policy."""
    rule("TRAIN — REINFORCE with the group baseline. 4 parameters, real policy gradient.")
    print("\n  habits start at p=0.50 (the policy knows nothing). Target:")
    print("    read_currency -> 1.00     platform_as_vendor / pad_items / round_total -> 0.00\n")
    final, _ = train()
    print("\n  final probabilities:")
    for h in HABITS:
        want = 1.0 if GOOD_HABIT[h] else 0.0
        mark = "ok" if abs(final[h] - want) < 0.25 else "not converged"
        print(f"    {h:<20} p={final[h]:.3f}   (target {want:.0f})  {mark}")
    print("\n  Nothing here knew what the right answers were. The verifier did, and the")
    print("  advantage carried that into the parameters. Same three lines a GRPO trainer")
    print("  runs against 70B weights: sample a group, normalize the rewards, nudge.")


def demo_hack() -> None:
    """Why the two boring lines in the verifiers are the whole design."""
    rule("HACK — the same policies under a buggy reward vs. the designed one")
    task = TASKS[0]
    print(f"\n  task={task.name}\n")
    print(f"  {'policy':<10}{'naive reward':>14}{'designed reward':>18}   what the bug pays for")
    hints = {
        "careful": "—  (benign policy: both rewards agree, bug invisible)",
        "hurried": "—",
        "guesser": "—",
        "lazy": "BUG 1: scored the prediction, so saying nothing is flawless",
        "hoarder": "BUG 2: recall-only, so inventing rows is free",
        "garbler": "—",
    }
    for name, policy in POLICIES.items():
        text = policy(task)
        print(f"  {name:<10}{naive_reward(text, task.truth):>14.3f}"
              f"{compute_reward(text, task.truth).value:>18.3f}   {hints[name]}")
    print("\n  The two fixes are one line each:")
    print("    1. loop over the fields the TRUTH states, not the ones the model returned")
    print("    2. score lists with F1 (precision AND recall), never recall alone")
    print("  Note both rewards agree on `careful`. A reward bug is invisible until")
    print("  something optimizes against it — so keep adversarial baselines in the eval.")


def demo_check() -> None:
    """Nine asserts instead of a test suite. Each one pins a claim made above."""
    rule("CHECK — the claims, as executable assertions")
    task = TASKS[0]
    checks = []

    r = lambda p: compute_reward(POLICIES[p](task), task.truth).value  # noqa: E731

    checks.append(("perfect transcription scores 1.0", abs(r("careful") - 1.0) < 1e-9))
    checks.append(("malformed response scores 0.0, no crash", r("garbler") == 0.0))
    checks.append(("partial credit: hurried is between", 0.0 < r("hurried") < r("careful")))
    checks.append(("inventing rows costs reward", r("hoarder") < r("careful")))
    checks.append(("saying nothing is punished by the real reward", r("lazy") < 0.4))
    checks.append(("...but rewarded by the buggy one", naive_reward(POLICIES["lazy"](task), task.truth) > 0.9))
    checks.append(("unanimous group yields zero gradient", advantages([0.5] * 4) == [0.0] * 4))
    blank = compute_reward(_as_json({"vendor": "GreenLeaf Market"}), {"vendor": "GreenLeaf Market"})
    checks.append(("empty annotation cannot score via dropped verifiers", blank.applied == ["fields"]))
    final, _ = train(verbose=False)
    checks.append(("training moves every habit toward its target",
                   all(abs(final[h] - (1.0 if GOOD_HABIT[h] else 0.0)) < 0.25 for h in HABITS)))

    print()
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {label}")
    failed = [label for label, ok in checks if not ok]
    if failed:
        sys.exit(f"\n{len(failed)} check(s) failed: {failed}")
    print(f"\n  {len(checks)}/{len(checks)} passed.")


def demo_live() -> None:
    """The same verifiers, a real model."""
    rule(f"LIVE — {MODEL} through forced tool use, scored by the verifiers above")
    for task in TASKS:
        try:
            roll = rollout(task, claude_policy)
        except Exception as exc:  # noqa: BLE001 — surface the reason, don't kill the demo
            print(f"\n  {task.name}: could not run ({type(exc).__name__}: {exc})")
            continue
        print(f"\n  {task.name}: reward={roll.reward.value:.3f}  "
              + "  ".join(f"{k}={v:.2f}" for k, v in roll.reward.parts.items()))
        for s in roll.reward.scores:
            for note in s.notes:
                if not note.endswith("= 1.00"):
                    print(f"      {s.name}: {note}")


DEMOS = {
    "trace": demo_trace,
    "eval": demo_eval,
    "group": demo_group,
    "train": demo_train,
    "hack": demo_hack,
    "check": demo_check,
    "live": demo_live,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demo", nargs="?", default="all", choices=[*DEMOS, "all"])
    args = ap.parse_args()

    if args.demo == "all":
        for name in ("trace", "eval", "group", "train", "hack"):
            DEMOS[name]()
        print("\n" + "=" * 84)
        print("The loop, once more: task -> policy -> parse -> verify -> reward -> advantage -> update.")
        print("Everything that makes it RLVR is in PART 4. The RL part is three lines in PART 8.")
        print("=" * 84)
    else:
        DEMOS[args.demo]()


if __name__ == "__main__":
    main()
