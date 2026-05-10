# Chad

*A self-improving generation system that learns from selection alone.*

---

## What Chad is

Chad is a lightweight system that helps a generator (drafter, linker, recommender — anything that produces candidates) get better over time without curated training data, without explicit feedback, and without the user writing rubrics.

The user does one thing: drops items into a bucket, and occasionally selects one. From those two acts, Chad infers what "good" means in this domain and steers future generation toward it.

## Core idea

Most "AI that learns from you" systems require curation: tagging, rating, rubric-writing. That work decays — humans don't sustain it. Chad replaces curation with **two acts the user already performs**:

- **Tossing** things into a bucket (cheap, low-judgment — anything that vaguely belongs)
- **Selecting** one when the moment requires (the actual judgment)

Selection is the signal. The bucket is the context. Tagging never happens.

## The field, not the list

Chad treats the bucket as a **topology**, not a list. Every item embeds to a point in semantic space. The bucket is a cloud. Selections are denser regions within the cloud. New drafts are probes into the field.

This reframes evaluation as **navigation**. Drafts aren't "good" or "bad" — they have a *location* relative to the field, and that location is what the system steers by.

## The three axes

Every draft gets scored on three independent dimensions:

- **Similar-to** — proximity to dense regions of bucket+selections (where the user has shown taste)
- **Shared-with** — degree of family membership in existing clusters (does it belong to a recognizable bunch)
- **Missing-from** — distance from existing density, weighted by reachability (does it explore an empty region the bucket doesn't yet cover)

The score is a 3D vector, not a scalar. This matters because:

- Similar-to alone collapses toward the mean — everything sounds the same
- Shared-with reveals which sub-bunch a draft joins
- Missing-from is exploration as a first-class citizen — the system can deliberately push into voids

Different generation requests live in different regions of the cube.

## Operations Chad supports

The user can issue requests like:

- *"Write one in the dense zone"* → high similar, high shared, low missing
- *"Write one like that cluster"* → high shared with a specific bunch
- *"Write into the void"* → high missing, see what emerges
- *"Bridge two clusters"* → equidistant from two dense regions
- *"Give me X drafts like the one with [vague description]"* → embed the vague description, find the nearest selection or cluster, generate near it

The last one is the killer feature: vague natural-language queries resolve to regions of the field, not to retrieved documents. The user describes the *vibe* and Chad finds where that lives, then generates there.

## Architecture

Three components plus a loop.

### 1. The bucket store

A flat append-only store. Each item:

```
{
  id,
  content,           # raw text or pointer
  embedding,         # vector
  source,            # where it came from (optional)
  added_at,
  selected_count,    # incremented when user picks it
  family_id          # cluster membership, recomputed periodically
}
```

No tags. No ratings. No rubric fields. Tossing is one function call; selecting is one increment.

### 2. The field

Derived from the bucket. Recomputed when the bucket changes meaningfully (e.g., N new items, or M new selections, or scheduled).

The field exposes:

- `density(point)` → scalar, how dense the bucket is at this location
- `clusters()` → list of family centroids and members
- `selection_gradient()` → direction in space from bucket-mean to selection-mean
- `voids()` → low-density regions reachable from existing clusters

Implementation: embeddings + a density estimator (KDE or nearest-neighbor density) + a clustering pass (HDBSCAN works well — handles arbitrary cluster counts and identifies outliers naturally).

### 3. The scorer

Given a candidate, return its 3D coordinates:

```
score(candidate) → (similar, shared, missing)

similar = density(candidate) projected along selection_gradient
shared  = max cluster_membership_strength across all clusters
missing = distance to nearest dense region, normalized by reachability
```

These are independent measurements. A draft can be high on all three (sits near a cluster's edge, in a region the gradient points toward, but in a thin part of the cluster) or low on all three (random noise far from anything).

### 4. The generation loop

The drafter is whatever model you wire in. Chad doesn't care. The loop:

```
request: "give me 5 drafts like [vague description]"
↓
resolve target region:
  - if vague description: embed it, find nearest cluster or selection
  - if cluster name: use that cluster's centroid
  - if "void": pick highest missing-from region within reachable space
  - if axis specification: use it directly
↓
generate N candidates near target region (drafter, biased by region context)
↓
score all candidates → 3D coordinates
↓
filter / rank by user's request shape
↓
return top K
↓
(when user selects one) → increment selected_count → field update queued
```

## The self-improvement mechanism

Chad improves in three ways, all automatic:

**1. The selection gradient sharpens.** Every time the user picks a draft (or picks an item from the bucket for any reason), the gradient from bucket-mean to selection-mean updates. Over time, this direction becomes a stable pointer at "what the user actually wants" — without the user ever describing it.

**2. The cluster structure matures.** As the bucket grows, clusters refine. New families emerge, old ones split or merge. The drafter inherits this — "write one like that cluster" gets more meaningful as clusters get sharper.

**3. The voids reveal themselves.** Voids are computed, not declared. As the bucket fills, the system identifies regions it *doesn't* cover, and exploration requests can target those. The user gets surprised by the system finding territory they hadn't articulated.

None of this requires the user to do anything beyond toss and select.

## Optional: the critic

For domains where generation quality benefits from internal iteration (e.g., LinkedIn posts, where one draft isn't enough), Chad can run a tight self-critique loop *inside* the generation step:

```
generate K candidates → score all → keep top → mutate/regenerate around winner → score again → repeat until budget
```

The critic isn't a separate model with a rubric. The critic *is* the field. Candidates compete by their 3D coordinates against the user's request shape. This collapses the "rubric problem" entirely — the field is the rubric, and the field updates itself.

## Optional: outcome signals

If the domain provides outcome data (engagement, click-through, reply rate, dwell time), Chad can incorporate it as a *post-hoc* signal: items that performed well get a boost in selection-equivalent weight. This doesn't replace selection — it augments it for domains where selection alone is sparse.

## Type 3: Self-diagnosis — Chad asking for help

Chad has three modes of improvement:

- **Type 1** — learns from you (interaction, selection patterns, preferences)
- **Type 2** — learns from the domain (work quality, reference comparison, outcome signals)
- **Type 3** — knows what it doesn't know, and asks for help fixing it

Type 3 is the one that makes Chad durable. Not self-tuning within existing parameters — but self-*diagnosing*. Chad observing its own behavior and surfacing "I think I have a structural problem here" as a legible statement you can act on in code.

Fully autonomous self-modification is fragile and untrustworthy. *Diagnosed + proposed + human-approved* changes compound safely. Chad stays out of the operational loop, you stay out of the generation loop, but you meet in the improvement loop — and only when Chad has something specific to say.

### What Chad watches

A periodic diagnostic pass looks for patterns in its own failures:

- Voids keep producing low-selection output → missing-from calculation may be finding noise, not genuine gaps
- One cluster dominates all generations → fit() weights for cluster requests are too sticky
- Vague queries keep resolving to the same few items → embedding model may be too coarse for this domain's vocabulary
- Selection rate dropping across all request types over time → bucket may need pruning, old items diluting the field
- fit() weights haven't moved despite meta-Chad running → meta-Chad's bucket may be too sparse to have signal yet

Each is a *diagnosable pattern* Chad detects without you. The move is surfacing it as a specific, actionable statement rather than silent degradation.

### The diagnostic report

```
Chad health check — week of [date]

FLAGGED:

void requests: 12 generated, 1 selected (8%). Baseline 23%.
  Possible cause: missing-from regions are low-reachability noise.
  Suggested fix: raise reachability floor in fit(void) from 0.1 → 0.3
  → approve / modify / ignore

cluster "technical-personal": no selections in 6 weeks.
  Possible cause: content drift, cluster is stale.
  Suggested fix: rebuild clusters, let this one dissolve if underpopulated.
  → approve / ignore

meta-Chad: insufficient selection data to tune fit() weights.
  Possible cause: too few generation sessions to build signal.
  Suggested fix: none yet — flag again in 2 weeks.
  → acknowledge
```

You read it, approve or modify, the change gets applied. You're in the *improvement loop*, not the generation loop — and only when Chad has something specific to say.

### Why this is the right boundary

Type 1 and Type 2 handle continuous tuning autonomously. Type 3 handles the moments when tuning isn't enough and the architecture itself needs a conversation. Chad doesn't silently degrade, doesn't silently self-patch in ways you can't see. It asks. You decide. The system stays legible.

This is also where Chad becomes a collaborator rather than a tool — it can say *I want to improve this specific thing* and you get in the code together and improve it.

## What Chad is NOT

- Not a tagging system. Tags never appear.
- Not a rating system. The user never rates.
- Not a recommendation engine. It generates, it doesn't retrieve.
- Not a fine-tuner. The drafter model is unchanged. Chad steers the drafter's prompt context with field information.
- Not a chatbot. It's a generation primitive other systems wrap.

## Pseudocode

```python
# === Bucket operations ===

def toss(content, source=None):
    """User adds something to the bucket. No judgment, no tags."""
    embedding = embed(content)
    bucket.append({
        id: uuid(),
        content: content,
        embedding: embedding,
        source: source,
        added_at: now(),
        selected_count: 0,
        family_id: None  # assigned on next field rebuild
    })
    maybe_rebuild_field()

def select(item_id):
    """User picks an item. The signal."""
    item = bucket.get(item_id)
    item.selected_count += 1
    maybe_rebuild_field()


# === Field derivation ===

def rebuild_field():
    """Recompute the topology. Called periodically."""
    embeddings = [item.embedding for item in bucket]
    selection_weights = [item.selected_count + 1 for item in bucket]

    field.density_estimator = kde(embeddings, weights=selection_weights)
    field.clusters = hdbscan(embeddings)
    field.selection_gradient = (
        weighted_mean(embeddings, selection_weights) -
        mean(embeddings)
    )
    field.voids = find_low_density_reachable_regions(field.density_estimator)

    for item, cluster_id in zip(bucket, field.clusters):
        item.family_id = cluster_id


# === Scoring ===

def score(candidate_embedding):
    """Return 3D coordinates in the field."""
    similar = project(candidate_embedding, field.selection_gradient) \
              * field.density_estimator(candidate_embedding)
    shared  = max(
        cluster_membership(candidate_embedding, c)
        for c in field.clusters
    )
    missing = distance_to_nearest_dense_region(
        candidate_embedding,
        field.density_estimator
    ) * reachability(candidate_embedding, field)

    return (similar, shared, missing)


# === Target resolution ===

def resolve_target(request):
    """Translate a natural-language or structured request into a region."""
    if request.kind == "vague_description":
        target_emb = embed(request.text)
        nearest = nearest_selection_or_cluster(target_emb)
        return Region(center=nearest.embedding, spread=nearest.spread)

    if request.kind == "cluster":
        c = field.clusters[request.cluster_id]
        return Region(center=c.centroid, spread=c.radius)

    if request.kind == "void":
        return Region(center=field.voids[0], spread=default_spread)

    if request.kind == "axis":
        # e.g., "high missing, low shared"
        return resolve_by_axis_constraints(request.constraints)


# === Generation loop ===

def generate(request, n=5, internal_iterations=3):
    target = resolve_target(request)
    candidates = []

    for _ in range(internal_iterations):
        # drafter is biased toward target region via prompt construction
        new_candidates = drafter(
            context=region_to_context(target),
            n=n * 2
        )
        scored = [(c, score(embed(c))) for c in new_candidates]
        candidates.extend(scored)

        # narrow target around current best
        top = sorted(candidates, key=fit(request))[-n:]
        target = tighten_region(target, [c for c, _ in top])

    return sorted(candidates, key=fit(request))[-n:]


def fit(request):
    """Returns a function that scores a (candidate, 3d_score) tuple
       against the request's preferred shape.

       Each request type defines a preferred *shape* in the 3D score
       space (similar, shared, missing). fit() returns a scalar telling
       the loop how well a candidate matches that shape.

       The pattern: each request type specifies (a) a target point or
       direction in score-space, and (b) which axes matter. The fitness
       is a weighted combination — high on axes that matter, penalized
       for distance from the target shape."""

    if request.kind == "vague_description":
        # User described a vibe. Target region is near a specific
        # selection or cluster. Want HIGH similar to that region,
        # MODERATE shared (belongs somewhere), LOW missing (not exploring).
        target_emb = request.resolved_target.center
        def f(candidate, score):
            similar, shared, missing = score
            proximity = 1 - distance(candidate.embedding, target_emb)
            return (0.6 * proximity) + (0.3 * shared) - (0.4 * missing)
        return f

    if request.kind == "cluster":
        # User wants one like a specific bunch. HIGH shared with that
        # specific cluster, similar matters less, missing should be low.
        cluster_id = request.cluster_id
        def f(candidate, score):
            similar, shared, missing = score
            cluster_membership = membership_in(candidate, cluster_id)
            return (0.7 * cluster_membership) + (0.2 * similar) - (0.3 * missing)
        return f

    if request.kind == "void":
        # User wants exploration. HIGH missing, but not so far it's
        # noise — needs some reachability to existing structure.
        def f(candidate, score):
            similar, shared, missing = score
            # penalize candidates with zero shared (totally disconnected)
            reachability_floor = max(shared, 0.1)
            return (0.7 * missing) * reachability_floor
        return f

    if request.kind == "bridge":
        # User wants something between two clusters. Equidistant from
        # both, with shared membership in neither dominant.
        c1, c2 = request.cluster_ids
        def f(candidate, score):
            similar, shared, missing = score
            d1 = distance_to_cluster(candidate, c1)
            d2 = distance_to_cluster(candidate, c2)
            balance = 1 - abs(d1 - d2) / (d1 + d2 + 1e-6)
            closeness = 1 - ((d1 + d2) / 2)
            return (0.5 * balance) + (0.5 * closeness)
        return f

    if request.kind == "axis":
        # Direct specification: e.g., {"missing": "high", "shared": "low"}
        # Each constraint contributes linearly.
        def f(candidate, score):
            similar, shared, missing = score
            axes = {"similar": similar, "shared": shared, "missing": missing}
            total = 0
            for axis, preference in request.constraints.items():
                value = axes[axis]
                if preference == "high":
                    total += value
                elif preference == "low":
                    total += (1 - value)
                elif preference == "mid":
                    total += 1 - abs(value - 0.5) * 2
            return total / len(request.constraints)
        return f

    # Default: prefer high similar (safe, mean-seeking)
    return lambda candidate, score: score[0]

# NOTE: the weights above (0.6, 0.3, 0.4, etc.) are starting guesses.
# Worth adding a meta-Chad on top: a second Chad instance whose bucket
# is (request, candidates_returned, which_one_user_picked) tuples.
# Selection signal here = which weight configuration produced the
# candidate the user actually chose. Over time, the meta-Chad's field
# steers the weights themselves — the system tunes its own fit()
# without anyone hand-adjusting numbers. Chad watching Chad.


# === Type 3: Self-diagnosis ===

def run_diagnostic(baseline_window_days=30):
    """Periodic pass — Chad inspects its own behavior and flags structural issues.
       Returns a report for human review. Nothing changes without approval."""
    report = []

    # Check void selection rate vs baseline
    void_decisions = decisions.query(kind="void", days=baseline_window_days)
    void_rate = selection_rate(void_decisions)
    baseline_rate = selection_rate(decisions.query(days=baseline_window_days * 3))
    if void_rate < baseline_rate * 0.5:
        report.append({
            "flag": "void requests underperforming",
            "evidence": f"{void_rate:.0%} vs baseline {baseline_rate:.0%}",
            "possible_cause": "missing-from regions are low-reachability noise",
            "suggested_fix": "raise reachability floor in fit(void) from current → +0.2",
            "action": ["approve", "modify", "ignore"]
        })

    # Check cluster staleness
    for cluster in field.clusters:
        if days_since(cluster.last_selected_at) > 42:  # 6 weeks
            report.append({
                "flag": f"cluster '{cluster.label}' stale",
                "evidence": f"no selections in {days_since(cluster.last_selected_at)} days",
                "possible_cause": "content drift or cluster no longer relevant",
                "suggested_fix": "rebuild clusters, dissolve if underpopulated",
                "action": ["approve", "ignore"]
            })

    # Check if meta-Chad has enough signal to tune weights
    if meta_chad.bucket.selection_count() < 20:
        report.append({
            "flag": "meta-Chad signal too sparse to tune fit() weights",
            "evidence": f"only {meta_chad.bucket.selection_count()} selections recorded",
            "possible_cause": "not enough generation sessions yet",
            "suggested_fix": "none yet — flag again in 2 weeks",
            "action": ["acknowledge"]
        })

    # Check overall selection rate trend
    recent_rate = selection_rate(decisions.query(days=7))
    older_rate = selection_rate(decisions.query(days=30, offset=7))
    if recent_rate < older_rate * 0.7:
        report.append({
            "flag": "selection rate declining",
            "evidence": f"recent {recent_rate:.0%} vs prior {older_rate:.0%}",
            "possible_cause": "bucket aging, field drift, or drafter quality drop",
            "suggested_fix": "prune items older than 6 months with zero selections",
            "action": ["approve", "modify", "ignore"]
        })

    return report  # surfaced to user — nothing applied until approved
```

## Implementation notes

- **Embedding model**: any sentence-embedding model. Local works fine (e.g., a small sentence-transformer). Quality matters less than consistency — same model for bucket items and candidates.
- **Density estimator**: start with k-NN density (count neighbors within radius). KDE is fine but heavier. Either way, recompute lazily.
- **Clustering**: HDBSCAN is the default — no need to specify cluster count, handles noise. Recompute on schedule, not per-toss.
- **Storage**: SQLite + a vector column (sqlite-vss, or just store embeddings as blobs and load into numpy). Bucket of <100k items needs no real database.
- **Field rebuild cost**: O(n²) for some operations, but n is small (you're not curating the entire internet — you're curating a domain bucket). Even 10k items rebuilds in seconds. Cache aggressively.
- **Drafter integration**: Chad is agnostic. It hands the drafter a *region* (centroid + spread + nearby examples) and the drafter does whatever it does. The region is just context injection.

## How Chad lives inside other systems

Chad is a primitive. It plugs into anything that generates candidates and benefits from field-aware steering:

- **Perch (LinkedIn drafter)**: bucket = posts the user found interesting (own + others'). Selections = drafts published. Generation steered by field.
- **WikiAI (linker)**: bucket = file pairs the user has linked manually or implicitly co-edited. Selections = links the user kept. Generation = proposing new links biased toward the field's structure.
- **COS (daily digest)**: bucket = items the user opened/replied to from past digests. Selections = items that drove action. Generation = ranking today's candidates against the field.
- **Contact monitor (alert prioritizer)**: bucket = past alerts. Selections = ones that led to outreach. Generation = scoring new alerts.

Each system contributes its own bucket and queries Chad with its own request shapes. Chad's core code is shared.

## What's singular about Chad

Three things, named explicitly:

1. **Selection-as-signal with no tagging burden.** The user does what they were already doing.
2. **Field-as-rubric.** The standard is latent and discovered, never declared. It updates itself.
3. **Three-axis score.** Evaluation becomes navigation. Exploration is first-class. The system can deliberately push into territory the user hasn't articulated.

These compose into a primitive that gets sharper while you sleep, on data you accumulate by living.
