# Chad

> *"The most important feature distinguishing reinforcement learning from other types of learning is that it uses training information that evaluates the actions taken rather than instructs by giving correct actions."*
> — Richard S. Sutton & Andrew G. Barto, *Reinforcement Learning: An Introduction*, ch. 2

**A generation primitive that learns from selection alone.**

No tags. No ratings. No rubrics. You toss things into a bucket and occasionally pick one. Chad infers everything else.

---

## Why

Every "AI that learns from you" system asks for curation work that decays. Six months in, you've stopped tagging, and the system stops improving.

Selection is the signal you already produce. Bookmarks. Drafts you publish. Links you keep. Items you click. Chad treats only that as ground truth.

## How

The bucket isn't a list. It's a topology. Selections are denser regions in semantic space. New drafts are probes, scored on three independent axes:

- **similar-to** — proximity to where you've shown taste
- **shared-with** — family membership in existing clusters
- **missing-from** — distance into territory the bucket hasn't covered

Most systems optimize toward the mean and collapse. Chad can deliberately push into voids and see what comes back. Exploration is navigable, not accidental.

## What you ask it

You can ask:

```
"5 drafts like the one about the dog two weeks ago"
"Write into the void"
"Between the technical cluster and the personal cluster"
"High missing, low shared — surprise me"
```

Or you can ask nothing. Chad still improves. The only thing it asks *you* is when it spots a structural problem in itself:

> "Void requests selecting at 8% vs 23% baseline. Probable cause: missing-from regions are noise, not gaps. Suggested fix: raise reachability floor 0.1 → 0.3. Approve / modify / ignore?"

Vague descriptions resolve to regions of the field, not retrieved documents. The system understands *where* you mean.

## How it improves itself

Three loops, automatic:

- **Type 1** — your selections sharpen the gradient
- **Type 2** — domain outcomes refine the field
- **Type 3** — Chad diagnoses its own failures and asks you to approve fixes

Type 3 is the durability move. Chad doesn't silently degrade. You stay in the improvement loop. Out of the generation loop. The system stays legible.

A meta-Chad on top tunes Chad's own scoring weights using the same primitive. Chad watching Chad. Recursion bottoms out at "what did the user actually pick" — the only ground truth that ever existed.

## What's here

- **[chad.md](chad.md)** — full spec, architecture, pseudocode
- **[chad.py](chad.py)** — reference sketch, ~450 lines, runnable with your own embed and draft functions

Chad is a primitive, not an app. You bring three things:

1. A bucket — some corpus you care about
2. A drafter — any model that generates candidates
3. A selection surface — anything that captures your picks, even a CLI

Chad brings the field.

## The pattern, named

Any system with selection in its loop can self-improve — including a system whose job is improving other systems. The maintenance overhead that kills ambitious AI builds doesn't apply, because there's nothing to maintain. The system curates itself by being used.

## Lineage

Chad sits at the intersection of work older than itself. Worth naming the rooms it inherits from:

**Implicit feedback systems.** Susan Dumais and the IR crowd at Microsoft Research; Thorsten Joachims at Cornell, whose *Optimizing Search Engines using Clickthrough Data* (2002) is the canonical "selection is signal" paper. Chad's first move — taking selection as ground truth — is their move, applied to generation instead of retrieval.

**Exploration as first-class.** Rich Sutton's work on the explore/exploit tradeoff. The void axis is exploration made navigable rather than stochastic.

**Latent space as territory.** Hinton on distributed representations; LeCun on world models and the limits of explicit supervision. Chad treats the embedding space as a place you can describe, point at, and move through.

**Tools that compound with use.** Doug Engelbart's *Augmenting Human Intellect* (1962). Alan Kay on systems that adapt. Type 3 — the system asking for help — is Engelbart's framing of human-computer collaboration, applied at the diagnostic layer.

**Selection as the source of design.** Christopher Alexander, *A Pattern Language* and *The Nature of Order*. Not an AI lineage at all — and that's the point. The pattern is older than the field. Good structure emerges from selection, not declaration.

## Outside the field

Chad also inherits from rooms most ML papers don't cite:

**Eugene Gendlin** — *Focusing* and the felt sense. Meaning lives in places words haven't reached yet; you find it by pointing, not defining. The vague-description query is felt-sense retrieval.

**Carl Rogers** — person-centered therapy. Growth happens when you stop forcing the person toward external categories. Chad doesn't impose tags.

**Donald Winnicott** — the holding environment. Structure that holds space without demanding performance. The bucket as holding environment.

**Daniel Kahneman** — System 1 / System 2. Selection is System 1, fast and durable. Tagging is System 2 and decays. Chad bets on the cheap signal.

**Mihaly Csikszentmihalyi** — *flow*. Tools that stay out of the way during the doing, asking for input only when it serves the work. Chad's "ask nothing or ask something specific" contract is flow-aware design.

## Status

Spec complete. Reference implementation drafted, not battle-tested — math has rough edges, code is illustrative.

First instances shipping: COS, then Perch.

If the pattern resonates, fork it. The field is the artifact.

---

> *"It is how we choose what we do, and how we approach it, that will determine whether the sum of our days adds up to a formless blur, or to something resembling a work of art."*
> — Mihaly Csikszentmihalyi
