"""
chad.py — minimal runnable skeleton

No domain logic. Bring your own:
  - embed()   : str → list[float]
  - draft()   : context dict → list[str]
  - bucket    : a folder of .txt files or a list of strings to start

Install:
  pip install numpy scikit-learn hdbscan sqlite-utils
"""

import json
import math
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from sklearn.neighbors import KernelDensity
import hdbscan


# ── Storage ──────────────────────────────────────────────────────────────────

DB_PATH = Path("chad.db")


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS bucket (
            id TEXT PRIMARY KEY,
            content TEXT,
            embedding BLOB,
            source TEXT,
            added_at TEXT,
            selected_count INTEGER DEFAULT 0,
            family_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS decisions (
            id TEXT PRIMARY KEY,
            kind TEXT,
            context TEXT,
            choice TEXT,
            timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS reactions (
            id TEXT PRIMARY KEY,
            decision_id TEXT,
            reaction TEXT,
            magnitude REAL,
            timestamp TEXT
        );
    """)
    con.commit()
    return con


# ── Bucket operations ─────────────────────────────────────────────────────────

def toss(con, content: str, embed_fn: Callable, source: str = None):
    """Add an item to the bucket. No tags, no judgment."""
    embedding = embed_fn(content)
    con.execute(
        "INSERT INTO bucket VALUES (?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), content, _pack(embedding),
         source, _now(), 0, None)
    )
    con.commit()


def select(con, item_id: str):
    """User picks an item. The signal."""
    con.execute(
        "UPDATE bucket SET selected_count = selected_count + 1 WHERE id = ?",
        (item_id,)
    )
    con.commit()


def _load_bucket(con):
    rows = con.execute(
        "SELECT id, content, embedding, selected_count, family_id FROM bucket"
    ).fetchall()
    return [
        {"id": r[0], "content": r[1], "embedding": _unpack(r[2]),
         "selected_count": r[3], "family_id": r[4]}
        for r in rows
    ]


# ── Field ─────────────────────────────────────────────────────────────────────

@dataclass
class Field:
    embeddings: np.ndarray
    weights: np.ndarray
    clusters: object          # fitted HDBSCAN
    kde: object               # fitted KernelDensity
    selection_gradient: np.ndarray
    cluster_labels: np.ndarray


def rebuild_field(con) -> Optional[Field]:
    items = _load_bucket(con)
    if len(items) < 5:
        return None  # not enough data yet

    embs = np.array([i["embedding"] for i in items])
    weights = np.array([i["selected_count"] + 1 for i in items], dtype=float)
    weights /= weights.sum()

    kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(embs, sample_weight=weights)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, prediction_data=True)
    cluster_labels = clusterer.fit_predict(embs)

    # Update family_id in db
    for item, label in zip(items, cluster_labels):
        con.execute("UPDATE bucket SET family_id = ? WHERE id = ?", (int(label), item["id"]))
    con.commit()

    selection_gradient = (
        np.average(embs, weights=weights, axis=0) - embs.mean(axis=0)
    )

    return Field(
        embeddings=embs,
        weights=weights,
        clusters=clusterer,
        kde=kde,
        selection_gradient=selection_gradient,
        cluster_labels=cluster_labels
    )


# ── Scoring ───────────────────────────────────────────────────────────────────

def score(embedding: list, field: Field) -> tuple[float, float, float]:
    """Return (similar, shared, missing) — position in the field."""
    emb = np.array(embedding).reshape(1, -1)

    # similar: density projected along selection gradient
    density = math.exp(field.kde.score_samples(emb)[0])
    grad_norm = np.linalg.norm(field.selection_gradient) + 1e-8
    projection = float(np.dot(emb[0], field.selection_gradient) / grad_norm)
    similar = float(np.clip(density * (projection + 1) / 2, 0, 1))

    # shared: cluster membership strength via approximate_predict
    # returns (labels, strengths) — strength is membership confidence in assigned cluster
    labels, strengths = hdbscan.approximate_predict(field.clusters, emb)
    shared = float(strengths[0]) if labels[0] != -1 else 0.0

    # missing: inverse density, clipped to reachable space
    nearest_dist = float(np.min(np.linalg.norm(field.embeddings - emb, axis=1)))
    max_dist = float(np.max(np.linalg.norm(field.embeddings - field.embeddings.mean(axis=0), axis=1)))
    missing = float(np.clip(nearest_dist / (max_dist + 1e-8), 0, 1))

    return similar, shared, missing


# ── fit() — request shape → fitness function ──────────────────────────────────

def fit(request: dict, field: Field) -> Callable:
    """Returns a function (embedding, 3d_score) → float."""
    kind = request.get("kind")

    if kind == "vague_description":
        target = np.array(request["target_embedding"])
        target_norm = np.linalg.norm(target) + 1e-8
        def f(emb, sc):
            emb_arr = np.array(emb)
            # cosine similarity, mapped to [0, 1]
            cos_sim = float(np.dot(emb_arr, target) / (np.linalg.norm(emb_arr) * target_norm + 1e-8))
            proximity = (cos_sim + 1) / 2
            similar, shared, missing = sc
            return (0.6 * proximity) + (0.3 * shared) - (0.4 * missing)
        return f

    if kind == "cluster":
        cid = request["cluster_id"]
        def f(emb, sc):
            similar, shared, missing = sc
            emb_arr = np.array(emb).reshape(1, -1)
            labels, strengths = hdbscan.approximate_predict(field.clusters, emb_arr)
            cluster_membership = float(strengths[0]) if labels[0] == cid else 0.0
            return (0.7 * cluster_membership) + (0.2 * similar) - (0.3 * missing)
        return f

    if kind == "void":
        def f(emb, sc):
            similar, shared, missing = sc
            reachability_floor = max(shared, 0.1)
            return (0.7 * missing) * reachability_floor
        return f

    if kind == "bridge":
        c1, c2 = request["cluster_ids"]
        def f(emb, sc):
            emb_arr = np.array(emb).reshape(1, -1)
            c1_center = field.embeddings[field.cluster_labels == c1].mean(axis=0)
            c2_center = field.embeddings[field.cluster_labels == c2].mean(axis=0)
            d1 = float(np.linalg.norm(emb_arr - c1_center))
            d2 = float(np.linalg.norm(emb_arr - c2_center))
            balance = 1 - abs(d1 - d2) / (d1 + d2 + 1e-8)
            closeness = 1 - (d1 + d2) / 2
            return (0.5 * balance) + (0.5 * closeness)
        return f

    if kind == "axis":
        axes_map = {"similar": 0, "shared": 1, "missing": 2}
        constraints = request["constraints"]
        def f(emb, sc):
            total = 0
            for axis, pref in constraints.items():
                v = sc[axes_map[axis]]
                if pref == "high":   total += v
                elif pref == "low":  total += (1 - v)
                elif pref == "mid":  total += 1 - abs(v - 0.5) * 2
            return total / len(constraints)
        return f

    # Default: prefer high similar
    return lambda emb, sc: sc[0]

    # need to set how the scoring function for "how well does this
    # candidate match the request shape" depends on which request
    # types you actually use first


# ── Generation loop ───────────────────────────────────────────────────────────

def generate(
    request: dict,
    field: Field,
    embed_fn: Callable,
    draft_fn: Callable,
    n: int = 5,
    internal_iterations: int = 3
) -> list[dict]:
    """
    Generate n candidates steered toward the request's region.
    draft_fn receives a context dict and returns a list of strings.
    """
    fit_fn = fit(request, field)
    candidates = []

    # Build region context for drafter
    context = _region_context(request, field)

    for _ in range(internal_iterations):
        drafts = draft_fn(context=context, n=n * 2)
        for draft in drafts:
            emb = embed_fn(draft)
            sc = score(emb, field)
            candidates.append({
                "content": draft,
                "embedding": emb,
                "score": sc,
                "fitness": fit_fn(emb, sc)
            })

        # Tighten context around current best
        top = sorted(candidates, key=lambda c: c["fitness"], reverse=True)[:n]
        context["examples"] = [c["content"] for c in top]

    return sorted(candidates, key=lambda c: c["fitness"], reverse=True)[:n]


def _region_context(request: dict, field: Field) -> dict:
    """Translate request into context dict for the drafter."""
    kind = request.get("kind", "vague_description")
    context = {"kind": kind, "examples": []}

    if kind == "cluster" and field is not None:
        cid = request.get("cluster_id", 0)
        mask = field.cluster_labels == cid
        if mask.any():
            # Find bucket items in this cluster — drafter gets them as examples
            context["cluster_id"] = cid
    elif kind == "void":
        context["instruction"] = "explore unexpected territory, avoid familiar patterns"
    elif kind == "vague_description":
        context["description"] = request.get("text", "")

    return context


# ── Type 3: Self-diagnosis ────────────────────────────────────────────────────

def run_diagnostic(con, field: Field, baseline_window_days: int = 30) -> list[dict]:
    """
    Chad inspects its own behavior and returns flagged issues.
    Nothing changes without human approval — this only returns the report.
    """
    report = []
    now = datetime.utcnow()

    def _selection_rate(rows):
        if not rows:
            return 0.0
        total = len(rows)
        reacted = con.execute(
            f"SELECT COUNT(*) FROM reactions WHERE decision_id IN "
            f"({','.join(['?']*total)}) AND reaction='acted_on'",
            [r[0] for r in rows]
        ).fetchone()[0]
        return reacted / total

    void_rows = con.execute(
        "SELECT id FROM decisions WHERE kind='void' AND timestamp > ?",
        ((now - timedelta(days=baseline_window_days)).isoformat(),)
    ).fetchall()

    all_rows = con.execute(
        "SELECT id FROM decisions WHERE timestamp > ?",
        ((now - timedelta(days=baseline_window_days * 3)).isoformat(),)
    ).fetchall()

    void_rate = _selection_rate(void_rows)
    baseline_rate = _selection_rate(all_rows)

    if void_rows and void_rate < baseline_rate * 0.5:
        report.append({
            "flag": "void requests underperforming",
            "evidence": f"{void_rate:.0%} selected vs baseline {baseline_rate:.0%}",
            "possible_cause": "missing-from regions are low-reachability noise",
            "suggested_fix": "raise reachability floor in fit(void) from 0.1 → 0.3",
            "action": ["approve", "modify", "ignore"]
        })

    # Cluster staleness
    if field:
        unique_labels = set(field.cluster_labels) - {-1}
        for cid in unique_labels:
            last = con.execute(
                "SELECT MAX(timestamp) FROM reactions r "
                "JOIN decisions d ON r.decision_id = d.id "
                "WHERE d.context LIKE ? AND r.reaction='acted_on'",
                (f'%"cluster_id": {cid}%',)
            ).fetchone()[0]
            if last is None or (now - datetime.fromisoformat(last)).days > 42:
                report.append({
                    "flag": f"cluster {cid} stale",
                    "evidence": f"no selections in 42+ days",
                    "possible_cause": "content drift or cluster no longer relevant",
                    "suggested_fix": "rebuild field, dissolve cluster if underpopulated",
                    "action": ["approve", "ignore"]
                })

    # Overall selection rate trend
    recent_rows = con.execute(
        "SELECT id FROM decisions WHERE timestamp > ?",
        ((now - timedelta(days=7)).isoformat(),)
    ).fetchall()
    older_rows = con.execute(
        "SELECT id FROM decisions WHERE timestamp BETWEEN ? AND ?",
        ((now - timedelta(days=30)).isoformat(), (now - timedelta(days=7)).isoformat())
    ).fetchall()

    recent_rate = _selection_rate(recent_rows)
    older_rate = _selection_rate(older_rows)

    if older_rate > 0 and recent_rate < older_rate * 0.7:
        report.append({
            "flag": "selection rate declining",
            "evidence": f"recent 7d: {recent_rate:.0%} vs prior 30d: {older_rate:.0%}",
            "possible_cause": "bucket aging, field drift, or drafter quality drop",
            "suggested_fix": "prune items older than 6 months with zero selections",
            "action": ["approve", "modify", "ignore"]
        })

    return report


# ── Utilities ─────────────────────────────────────────────────────────────────

def _pack(embedding: list) -> bytes:
    return np.array(embedding, dtype=np.float32).tobytes()

def _unpack(blob: bytes) -> list:
    return np.frombuffer(blob, dtype=np.float32).tolist()

def _now() -> str:
    return datetime.utcnow().isoformat()


# ── Minimal CLI demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Wire in your own embed_fn and draft_fn below, then run:

      python chad.py load <folder>          — toss .txt files into bucket
      python chad.py generate <description> — generate candidates
      python chad.py diagnose               — run Type 3 health check
    """
    import sys

    # ── REPLACE THESE WITH YOUR OWN ──────────────────────────────────────────
    # embed_fn: str → list[float]
    #   e.g. sentence-transformers, Ollama embeddings, OpenAI embeddings
    # draft_fn: (context: dict, n: int) → list[str]
    #   e.g. Ollama generate, Claude API, any local or remote LLM
    #
    # The stubs below let you run `load` and `diagnose` without a drafter
    # but `generate` requires both. Replace before using generate.

    def embed_fn(text: str) -> list:
        # Example using sentence-transformers (uncomment to use):
        # from sentence_transformers import SentenceTransformer
        # model = SentenceTransformer("all-MiniLM-L6-v2")
        # return model.encode(text).tolist()
        raise NotImplementedError("Wire in your embedding model")

    def draft_fn(context: dict, n: int) -> list:
        # Example using Ollama (uncomment to use):
        # import ollama
        # prompt = build_prompt_from_context(context)
        # return [ollama.generate(model="llama3", prompt=prompt)["response"]
        #         for _ in range(n)]
        raise NotImplementedError("Wire in your drafter")
    # ─────────────────────────────────────────────────────────────────────────

    con = init_db()

    cmd = sys.argv[1] if len(sys.argv) > 1 else None

    if cmd == "load":
        folder = Path(sys.argv[2])
        files = list(folder.glob("*.txt"))
        for f in files:
            toss(con, f.read_text(), embed_fn, source=str(f))
        print(f"Loaded {len(files)} items into bucket.")

    elif cmd == "generate":
        description = " ".join(sys.argv[2:]) or "something interesting"
        f = rebuild_field(con)
        if f is None:
            print("Bucket too small — toss in at least 5 items first.")
            sys.exit(1)

        target_emb = embed_fn(description)
        request = {"kind": "vague_description", "text": description,
                   "target_embedding": target_emb}
        results = generate(request, f, embed_fn, draft_fn, n=3)

        for i, r in enumerate(results):
            sim, sha, mis = r["score"]
            print(f"\n[{i+1}] similar={sim:.2f} shared={sha:.2f} missing={mis:.2f}")
            print(r["content"])
            keep = input("Keep this one? (y/n): ").strip().lower()
            if keep == "y":
                toss(con, r["content"], embed_fn, source="generated")
                item_id = con.execute(
                    "SELECT id FROM bucket ORDER BY added_at DESC LIMIT 1"
                ).fetchone()[0]
                select(con, item_id)
                print("Added to bucket and marked as selected.")

    elif cmd == "diagnose":
        f = rebuild_field(con)
        report = run_diagnostic(con, f)
        if not report:
            print("Chad is healthy. No flags.")
        for item in report:
            print(f"\nFLAG: {item['flag']}")
            print(f"  Evidence: {item['evidence']}")
            print(f"  Cause:    {item['possible_cause']}")
            print(f"  Fix:      {item['suggested_fix']}")
            print(f"  Actions:  {', '.join(item['action'])}")

    else:
        print("Usage:")
        print("  python chad.py load <folder>          — toss .txt files into bucket")
        print("  python chad.py generate <description> — generate candidates")
        print("  python chad.py diagnose               — run Type 3 health check")
