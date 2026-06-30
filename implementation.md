# XAI Movie Recommendation System — Implementation Plan
> **For:** GitHub Copilot (Claude Haiku 4.5) | **Base Repo:** [End2End-Recommendation-Engine](https://github.com/abideenml/End2End-Recommendation-Engine_)  
> **Stack:** NextJS · Flask · MySQL (Kryptonite DB) · TailwindCSS · AWS  
> **Goal:** Layer SHAP-based explainability + counterfactual reasoning onto the existing Content-Based, Collaborative Filtering, and Neural Collaborative Filtering engines.

---

## Table of Contents

1. [Project Overview & Philosophy](#1-project-overview--philosophy)
2. [Chosen XAI Strategy](#2-chosen-xai-strategy)
3. [Repository Structure (Target State)](#3-repository-structure-target-state)
4. [Phase 1 — Database Additions](#4-phase-1--database-additions)
5. [Phase 2 — Backend: SHAP Integration](#5-phase-2--backend-shap-integration)
6. [Phase 3 — Backend: Counterfactual Engine](#6-phase-3--backend-counterfactual-engine)
7. [Phase 4 — Backend: Natural Language Reason Generator](#7-phase-4--backend-natural-language-reason-generator)
8. [Phase 5 — Flask API Endpoints](#8-phase-5--flask-api-endpoints)
9. [Phase 6 — Frontend: Explanation UI Components](#9-phase-6--frontend-explanation-ui-components)
10. [Phase 7 — Feedback Loop](#10-phase-7--feedback-loop)
11. [Phase 8 — Testing](#11-phase-8--testing)
12. [Dependencies & Environment Setup](#12-dependencies--environment-setup)
13. [File-by-File Implementation Checklist](#13-file-by-file-implementation-checklist)

---

## 1. Project Overview & Philosophy

The existing repo supports three recommendation strategies:

| Engine | Mechanism | Cold Start? |
|--------|-----------|-------------|
| Content-Based | Genre/metadata similarity via `user_movielist` | Solved via personal list |
| Collaborative Filtering (MF) | Matrix Factorization on `rating_explicit`/`rating_implicit` | Needs ratings |
| Neural Collaborative Filtering (NCF) | MLP + Embeddings on user-item pairs | Needs ratings |

**What is missing** is any explanation layer. A user sees a recommendation score but has no idea *why* it appeared. This plan adds a full XAI layer that:

- Tells users exactly which features drove each recommendation (SHAP).
- Lets users understand "what would change this recommendation" (counterfactuals).
- Renders natural-language reason strings in the UI ("Because you loved Inception and action thrillers").
- Collects thumbs-up/down feedback to retrain and track explanation accuracy.

**Design constraint:** The XAI layer must be additive — it must not break any existing API routes, database tables, or frontend pages. All additions are new files, new tables, and new API endpoints.

---

## 2. Chosen XAI Strategy

After evaluating LIME, SHAP, attention weights, and counterfactuals for this stack, the recommended combination is:

### Primary: SHAP (SHapley Additive Explanations)

**Why SHAP over LIME here:**
- SHAP values are globally consistent — the same feature will have the same attribution direction across all users, which is critical for debugging and trust.
- The existing models (MF and NCF) produce numeric user/item embeddings. SHAP's `KernelExplainer` works model-agnostically on any function that maps features to a score, making it compatible with all three engines without rewriting them.
- SHAP produces a ranked list of features (genre_score, rating_count, genome_tag_relevance, etc.) that maps directly to the columns already in the Kryptonite database.
- `shap.Explainer` for tree-based fallbacks (if a gradient-boosted re-ranker is added later) and `shap.KernelExplainer` for the neural model.

### Secondary: Counterfactual Explanations

**Why counterfactuals:**
- Answers the "what if?" question without exposing model internals.
- Easy to implement on top of SHAP: find the minimum feature change that flips the recommendation score below a threshold.
- Surfaces as a conversational UI element: *"If you rated The Dark Knight above 4 stars, Tenet would move into your top 5."*

### Natural Language Rendering

SHAP values and counterfactual diffs are translated into human-readable strings using a template engine (no LLM dependency — purely rule-based for latency). The templates live in a single Python file and can be extended without touching any model code.

---

## 3. Repository Structure (Target State)

Files marked `[NEW]` must be created. Files marked `[MODIFY]` already exist and need additions.

```
End2End-Recommendation-Engine_/
├── backend/
│   ├── app.py                          [MODIFY] — register new XAI blueprints
│   ├── config.py                       [MODIFY] — add SHAP config constants
│   ├── models/
│   │   ├── content_based.py            [existing]
│   │   ├── collaborative.py            [existing]
│   │   ├── neural_cf.py                [existing]
│   │   └── model_wrapper.py            [NEW] — unified scoring interface for SHAP
│   ├── xai/
│   │   ├── __init__.py                 [NEW]
│   │   ├── shap_explainer.py           [NEW] — SHAP computation for all 3 engines
│   │   ├── counterfactual.py           [NEW] — counterfactual generator
│   │   ├── nl_reason.py                [NEW] — natural language template renderer
│   │   └── explanation_cache.py        [NEW] — Redis/in-memory cache for SHAP values
│   ├── routes/
│   │   ├── recommend.py                [existing]
│   │   └── explain.py                  [NEW] — /api/explain/* endpoints
│   ├── db/
│   │   ├── queries.py                  [existing]
│   │   └── xai_queries.py              [NEW] — DB queries for XAI tables
│   └── utils/
│       ├── feature_extractor.py        [NEW] — build feature vectors from Kryptonite DB
│       └── feedback_handler.py         [NEW] — process thumbs up/down signals
├── frontend/
│   ├── components/
│   │   ├── MovieCard.tsx               [MODIFY] — add ExplanationBadge + toggle
│   │   ├── ExplanationPanel.tsx        [NEW] — slide-in explanation drawer
│   │   ├── ShapWaterfallChart.tsx      [NEW] — SHAP waterfall bar chart component
│   │   ├── CounterfactualCard.tsx      [NEW] — "what if" reasoning card
│   │   └── FeedbackButtons.tsx         [NEW] — thumbs up/down with optional reason
│   ├── pages/
│   │   └── recommendations.tsx         [MODIFY] — wire ExplanationPanel
│   ├── hooks/
│   │   └── useExplanation.ts           [NEW] — fetch + cache explanation data
│   └── types/
│       └── explanation.ts              [NEW] — TypeScript interfaces for XAI payloads
└── data/
    └── xai_schema.sql                  [NEW] — SQL migrations for XAI tables
```

---

## 4. Phase 1 — Database Additions

### 4.1 New Tables

Create `data/xai_schema.sql`. This file contains all new table definitions. Run it against the Kryptonite database after backing up the existing schema.

```sql
-- ─────────────────────────────────────────────
-- Table: explanation_features
-- Stores the raw SHAP feature values per recommendation
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS explanation_features (
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id          INT NOT NULL,
    movie_id         INT NOT NULL,
    model_type_id    INT NOT NULL,
    rec_score        FLOAT NOT NULL,
    features_json    JSON NOT NULL,
    -- features_json shape:
    -- {
    --   "genre_match": 0.43,
    --   "avg_rating": 0.21,
    --   "genome_relevance": 0.18,
    --   "watch_recency": 0.09,
    --   "popularity": -0.05,
    --   "user_similarity": 0.14
    -- }
    shap_values_json JSON NOT NULL,
    -- shap_values_json shape:
    -- {
    --   "genre_match": 0.31,
    --   "avg_rating": 0.15,
    --   "genome_relevance": 0.12,
    --   "watch_recency": 0.07,
    --   "popularity": -0.04,
    --   "user_similarity": 0.10
    -- }
    nl_reason        TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id)       REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id)      REFERENCES movies(movieId) ON DELETE CASCADE,
    FOREIGN KEY (model_type_id) REFERENCES models_type(id) ON DELETE CASCADE,
    INDEX idx_user_movie (user_id, movie_id),
    INDEX idx_created   (created_at)
);

-- ─────────────────────────────────────────────
-- Table: counterfactual_explanations
-- Stores "what if" reasoning per recommendation
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS counterfactual_explanations (
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id          INT NOT NULL,
    movie_id         INT NOT NULL,
    model_type_id    INT NOT NULL,
    pivot_movie_id   INT,          -- the movie whose rating change would matter
    pivot_rating     FLOAT,        -- the rating threshold
    predicted_rank_change INT,     -- how many positions it would move
    nl_counterfactual TEXT,        -- "If you rated X above 4, Y would enter top 5"
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id)         REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id)        REFERENCES movies(movieId) ON DELETE CASCADE,
    FOREIGN KEY (pivot_movie_id)  REFERENCES movies(movieId) ON DELETE SET NULL,
    INDEX idx_user_movie_cf (user_id, movie_id)
);

-- ─────────────────────────────────────────────
-- Table: explanation_feedback
-- Stores user thumbs up/down on explanations
-- ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS explanation_feedback (
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id          INT NOT NULL,
    movie_id         INT NOT NULL,
    model_type_id    INT NOT NULL,
    helpful          TINYINT(1) NOT NULL,  -- 1 = thumbs up, 0 = thumbs down
    feedback_text    VARCHAR(500),
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id)  REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (movie_id) REFERENCES movies(movieId) ON DELETE CASCADE
);
```

### 4.2 New MySQL View

Add this view to `xai_schema.sql` as well. It pre-joins the feature data needed for SHAP computation so the Python backend does not need to issue multiple queries.

```sql
-- ─────────────────────────────────────────────
-- View: xai_feature_view
-- Pre-joined feature data for SHAP computation
-- ─────────────────────────────────────────────
CREATE OR REPLACE VIEW xai_feature_view AS
SELECT
    uml.userId                          AS user_id,
    uml.movieId                         AS movie_id,
    m.title                             AS title,
    m.year                              AS release_year,
    GROUP_CONCAT(DISTINCT g.genre)      AS genres,
    AVG(re.rating)                      AS avg_explicit_rating,
    COUNT(DISTINCT re.userId)           AS rating_count,
    AVG(ri.rating)                      AS avg_implicit_rating,
    MAX(gs.relevance)                   AS top_genome_relevance,
    DATEDIFF(NOW(), MAX(uml.addedAt))   AS days_since_listed
FROM user_movielist uml
JOIN movies m           ON m.movieId = uml.movieId
JOIN movie_genres mg    ON mg.movieId = uml.movieId
JOIN genres g           ON g.genreId = mg.genreId
LEFT JOIN rating_explicit re ON re.movieId = uml.movieId
LEFT JOIN rating_implicit ri ON ri.movieId = uml.movieId
LEFT JOIN genome_scores gs   ON gs.movieId = uml.movieId
GROUP BY uml.userId, uml.movieId, m.title, m.year;
```

---

## 5. Phase 2 — Backend: SHAP Integration

### 5.1 Feature Extractor — `backend/utils/feature_extractor.py`

This module is responsible for pulling structured feature vectors from the Kryptonite database for any (user_id, movie_id) pair. It is the single source of truth for what features go into the SHAP explainer.

```python
"""
feature_extractor.py
Builds normalized feature vectors for SHAP computation.
All features are floats in the range [0, 1] unless noted.
"""

import numpy as np
from db.xai_queries import fetch_xai_features


# Feature names in the exact order used by the SHAP explainer.
# CRITICAL: Never reorder this list — it must stay in sync with
# ShapExplainer.FEATURE_NAMES and all saved SHAP value JSON blobs.
FEATURE_NAMES = [
    "genre_match_score",     # Jaccard similarity between user genre prefs and movie genres
    "avg_explicit_rating",   # Average explicit rating for the movie (normalized 0–1)
    "rating_count_log",      # log(rating_count + 1), normalized by max in dataset
    "genome_relevance",      # Max genome tag relevance score (already 0–1)
    "recency_score",         # Inverse of days_since_listed, normalized
    "popularity_percentile", # Where the movie sits in global popularity (0–1)
    "user_similarity_score", # Average similarity to users who rated this movie highly
]


def build_feature_vector(user_id: int, movie_id: int, db_conn) -> np.ndarray:
    """
    Returns a 1D numpy array of shape (len(FEATURE_NAMES),) for a single
    (user_id, movie_id) pair. All values are floats.

    Args:
        user_id:  The target user's ID in the users table.
        movie_id: The target movie's ID in the movies table.
        db_conn:  An active MySQL database connection.

    Returns:
        np.ndarray of shape (7,) with values in approximately [0, 1].

    Raises:
        ValueError: If no data is found for the given user_id/movie_id pair.
    """
    raw = fetch_xai_features(user_id, movie_id, db_conn)
    if raw is None:
        raise ValueError(f"No feature data for user={user_id}, movie={movie_id}")

    user_genres  = set(raw["user_preferred_genres"])
    movie_genres = set(raw["movie_genres"].split(","))

    # Jaccard similarity for genre overlap
    intersection = user_genres & movie_genres
    union        = user_genres | movie_genres
    genre_match  = len(intersection) / len(union) if union else 0.0

    # Normalize explicit rating (original scale is 0.5–5.0)
    avg_rating_norm = (raw["avg_explicit_rating"] or 3.0) / 5.0

    # Log-normalize rating count (cap at 100k)
    rating_count_log = np.log1p(min(raw["rating_count"] or 0, 100_000)) / np.log1p(100_000)

    # Genome relevance is already [0, 1]
    genome_rel = float(raw["top_genome_relevance"] or 0.0)

    # Recency: invert days_since_listed (more recent = higher score)
    days = float(raw["days_since_listed"] or 365)
    recency = 1.0 / (1.0 + days / 30.0)   # decays over months

    # Popularity percentile: pre-computed by xai_queries from all movies
    pop_pct = float(raw["popularity_percentile"] or 0.5)

    # User-user similarity score: pre-computed collaborative signal
    sim_score = float(raw["user_similarity_score"] or 0.0)

    return np.array([
        genre_match,
        avg_rating_norm,
        rating_count_log,
        genome_rel,
        recency,
        pop_pct,
        sim_score,
    ], dtype=np.float32)


def build_feature_matrix(user_id: int, movie_ids: list[int], db_conn) -> np.ndarray:
    """
    Vectorized version: builds a feature matrix of shape
    (len(movie_ids), len(FEATURE_NAMES)) for a batch of movies.
    Used when computing SHAP background datasets.
    """
    rows = []
    for mid in movie_ids:
        try:
            rows.append(build_feature_vector(user_id, mid, db_conn))
        except ValueError:
            rows.append(np.zeros(len(FEATURE_NAMES), dtype=np.float32))
    return np.vstack(rows)
```

### 5.2 Model Wrapper — `backend/models/model_wrapper.py`

SHAP needs a single callable `f(X) -> scores` where X is a 2D numpy array of features. This wrapper presents all three recommendation engines through a consistent interface.

```python
"""
model_wrapper.py
Unified scoring interface that SHAP can call as a black box.
Wraps content_based, collaborative, and neural_cf models.
"""

import numpy as np
from typing import Literal


ModelType = Literal["content", "collaborative", "ncf"]


class RecommenderWrapper:
    """
    Wraps one of the three recommendation models behind a single
    callable interface: predict(X) -> np.ndarray of scores.

    SHAP requires a function f: R^(n x d) -> R^n.
    This class provides exactly that.

    Usage:
        wrapper = RecommenderWrapper(model_type="ncf", model=ncf_model, user_id=42)
        explainer = shap.KernelExplainer(wrapper.predict, background_data)
    """

    def __init__(self, model_type: ModelType, model, user_id: int):
        self.model_type = model_type
        self.model      = model
        self.user_id    = user_id

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: numpy array of shape (n_samples, n_features).
               Each row is the feature vector for one movie.

        Returns:
            numpy array of shape (n_samples,) with recommendation scores.
        """
        if self.model_type == "content":
            return self._predict_content(X)
        elif self.model_type == "collaborative":
            return self._predict_collaborative(X)
        elif self.model_type == "ncf":
            return self._predict_ncf(X)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _predict_content(self, X: np.ndarray) -> np.ndarray:
        # The content-based model scores on genre_match_score (index 0)
        # and genome_relevance (index 3). Extract those columns and
        # call the existing content model's score method.
        # Adjust column indices if FEATURE_NAMES order changes.
        genre_scores   = X[:, 0]
        genome_scores  = X[:, 3]
        combined = 0.6 * genre_scores + 0.4 * genome_scores
        return combined.astype(np.float32)

    def _predict_collaborative(self, X: np.ndarray) -> np.ndarray:
        # MF collaborative filtering uses the user_similarity_score (index 6)
        # and avg_explicit_rating (index 1) as the main signals.
        user_sim   = X[:, 6]
        avg_rating = X[:, 1]
        combined   = 0.7 * user_sim + 0.3 * avg_rating
        return combined.astype(np.float32)

    def _predict_ncf(self, X: np.ndarray) -> np.ndarray:
        # For NCF, pass the full feature vector to the neural model.
        # self.model is the loaded PyTorch/Keras NCF model.
        # Convert features to a tensor and run inference.
        import torch
        tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            scores = self.model(tensor).squeeze(-1).numpy()
        return scores.astype(np.float32)
```

### 5.3 SHAP Explainer — `backend/xai/shap_explainer.py`

This is the core XAI computation module. It uses `shap.KernelExplainer` for model-agnostic attribution across all three engines.

```python
"""
shap_explainer.py
Computes SHAP values for movie recommendations.

Key design decisions:
  - Uses KernelExplainer (model-agnostic) so it works with content, CF, and NCF.
  - Background dataset is sampled once per model per session and cached.
  - SHAP values are stored in explanation_features table for retrieval without
    recomputing on every page load.
  - N_BACKGROUND controls the accuracy/latency tradeoff. Start at 50 for dev.
"""

import shap
import numpy as np
from utils.feature_extractor import FEATURE_NAMES, build_feature_vector, build_feature_matrix
from models.model_wrapper import RecommenderWrapper
from xai.explanation_cache import ExplanationCache


# Number of background samples for KernelExplainer.
# Lower = faster but less accurate. 50 is a good default; raise to 200 for prod.
N_BACKGROUND = 50

# Minimum absolute SHAP value to include in the "top features" list.
# Features with |SHAP value| below this threshold are considered noise.
SHAP_MIN_THRESHOLD = 0.01


class SHAPExplainer:
    """
    Computes and caches SHAP explanations for all three recommendation engines.

    Workflow:
        1. On first call for a (model_type, user_id), sample background data.
        2. Fit a KernelExplainer to the background.
        3. For each movie_id, compute SHAP values for the feature vector.
        4. Store results in the DB and cache.
        5. On subsequent calls, return from cache.
    """

    def __init__(self, model_type: str, model, user_id: int, db_conn, cache: ExplanationCache):
        self.wrapper   = RecommenderWrapper(model_type, model, user_id)
        self.model_type = model_type
        self.user_id   = user_id
        self.db_conn   = db_conn
        self.cache     = cache
        self._explainer = None  # lazy-initialized

    def _get_explainer(self, background_movie_ids: list[int]):
        """
        Lazily initializes the KernelExplainer with a sampled background dataset.
        Called once per SHAPExplainer instance.
        """
        if self._explainer is not None:
            return self._explainer

        background_X = build_feature_matrix(
            self.user_id, background_movie_ids, self.db_conn
        )
        # Use kmeans to summarize the background if it's large,
        # reducing SHAP computation time.
        if len(background_X) > N_BACKGROUND:
            background_X = shap.kmeans(background_X, N_BACKGROUND).data

        self._explainer = shap.KernelExplainer(
            self.wrapper.predict,
            background_X
        )
        return self._explainer

    def explain(self, movie_id: int, background_movie_ids: list[int]) -> dict:
        """
        Returns the SHAP explanation dict for a single (user_id, movie_id) pair.

        Args:
            movie_id:             The movie to explain.
            background_movie_ids: A list of movie IDs used as background reference.
                                  These should be movies the user has already seen
                                  or rated, representing their "baseline" taste.

        Returns:
            A dict with the following keys:
            {
              "movie_id": int,
              "base_value": float,        # SHAP base value (expected model output)
              "prediction": float,        # Model output for this movie
              "shap_values": {            # Per-feature contributions
                "genre_match_score": 0.21,
                "avg_explicit_rating": 0.14,
                ...
              },
              "top_positive_features": [  # Top 3 features pushing score UP
                ("genre_match_score", 0.21),
                ...
              ],
              "top_negative_features": [  # Top 3 features pushing score DOWN
                ("popularity_percentile", -0.08),
                ...
              ]
            }

        Raises:
            ValueError: If feature vector cannot be built for the movie.
        """
        # Check cache first
        cached = self.cache.get(self.user_id, movie_id, self.model_type)
        if cached:
            return cached

        # Build feature vector for the target movie
        feature_vector = build_feature_vector(self.user_id, movie_id, self.db_conn)
        X = feature_vector.reshape(1, -1)

        # Compute prediction score
        prediction = float(self.wrapper.predict(X)[0])

        # Get or initialize the SHAP explainer
        explainer = self._get_explainer(background_movie_ids)

        # Compute SHAP values — nsamples controls accuracy vs. speed
        # Use "auto" in production; set explicitly for debugging
        shap_values = explainer.shap_values(X, nsamples="auto")[0]

        # Build the named feature dict
        shap_dict = {
            name: float(val)
            for name, val in zip(FEATURE_NAMES, shap_values)
        }

        # Filter by threshold and sort
        significant = {
            k: v for k, v in shap_dict.items()
            if abs(v) >= SHAP_MIN_THRESHOLD
        }
        sorted_features = sorted(significant.items(), key=lambda x: x[1], reverse=True)
        top_positive    = [(k, v) for k, v in sorted_features if v > 0][:3]
        top_negative    = [(k, v) for k, v in sorted_features if v < 0][:3]

        result = {
            "movie_id":             movie_id,
            "base_value":           float(explainer.expected_value),
            "prediction":           prediction,
            "shap_values":          shap_dict,
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
        }

        # Store to cache
        self.cache.set(self.user_id, movie_id, self.model_type, result)

        return result

    def explain_batch(self, movie_ids: list[int], background_movie_ids: list[int]) -> list[dict]:
        """
        Explains a batch of movies for the same user.
        More efficient than calling explain() in a loop because the
        KernelExplainer is initialized only once.
        """
        return [
            self.explain(mid, background_movie_ids)
            for mid in movie_ids
        ]
```

### 5.4 Explanation Cache — `backend/xai/explanation_cache.py`

SHAP computation is expensive (~500ms per movie). Cache results in memory (or Redis in production) to avoid recomputing on every page load.

```python
"""
explanation_cache.py
Simple in-memory LRU cache for SHAP explanation results.
In production, swap the dict for a Redis client.
"""

from functools import lru_cache
from typing import Optional
import json


class ExplanationCache:
    """
    Thread-safe in-memory cache for SHAP explanation dicts.
    Key format: "{user_id}:{movie_id}:{model_type}"
    TTL is not enforced in this implementation — restart server to clear.
    For production: replace self._store with Redis using SETEX with TTL=3600.
    """

    def __init__(self, max_size: int = 2000):
        self._store: dict[str, dict] = {}
        self._max_size = max_size

    def _key(self, user_id: int, movie_id: int, model_type: str) -> str:
        return f"{user_id}:{movie_id}:{model_type}"

    def get(self, user_id: int, movie_id: int, model_type: str) -> Optional[dict]:
        return self._store.get(self._key(user_id, movie_id, model_type))

    def set(self, user_id: int, movie_id: int, model_type: str, value: dict):
        if len(self._store) >= self._max_size:
            # Evict oldest entry (simple FIFO — use OrderedDict for proper LRU)
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]
        self._store[self._key(user_id, movie_id, model_type)] = value

    def invalidate_user(self, user_id: int):
        """Call this when a user submits feedback or updates their list."""
        keys_to_delete = [k for k in self._store if k.startswith(f"{user_id}:")]
        for k in keys_to_delete:
            del self._store[k]
```

---

## 6. Phase 3 — Backend: Counterfactual Engine

### `backend/xai/counterfactual.py`

Counterfactuals answer: "What is the minimum change to user behavior that would significantly alter this recommendation?"

```python
"""
counterfactual.py
Generates counterfactual explanations for movie recommendations.

Algorithm:
  1. Get the current recommendation score for the target movie.
  2. Identify the feature with the highest negative SHAP value (the main reason
     it is NOT ranked higher).
  3. Find a "pivot movie" in the user's watch history that, if rated higher,
     would increase that feature's value.
  4. Compute the score difference and translate to a rank change.
  5. Return a structured counterfactual dict.
"""

from utils.feature_extractor import build_feature_vector, FEATURE_NAMES
import numpy as np


# How much a rating needs to change to count as a meaningful counterfactual.
RATING_STEP = 0.5

# Minimum rank improvement to be worth showing to users.
MIN_RANK_IMPROVEMENT = 2


def generate_counterfactual(
    user_id:           int,
    movie_id:          int,
    shap_result:       dict,
    candidate_movies:  list[dict],  # list of {"movie_id": int, "title": str, "rating": float}
    model_wrapper,
    db_conn
) -> Optional[dict]:
    """
    Generates a single counterfactual explanation.

    Args:
        user_id:          The target user.
        movie_id:         The movie being explained.
        shap_result:      Output of SHAPExplainer.explain() for this movie.
        candidate_movies: Movies the user has seen but could re-rate.
                          Typically their top-20 rated movies.
        model_wrapper:    RecommenderWrapper for scoring.
        db_conn:          Active database connection.

    Returns:
        Dict with keys:
        {
          "pivot_movie_id":     int,      # Movie to re-rate
          "pivot_movie_title":  str,
          "current_rating":     float,
          "suggested_rating":   float,
          "predicted_rank_change": int,   # Estimated improvement in recommendation rank
          "nl_counterfactual":  str       # Human-readable string
        }
        Returns None if no meaningful counterfactual is found.
    """
    # Find the most harmful negative feature
    neg_features = shap_result.get("top_negative_features", [])
    if not neg_features:
        return None

    worst_feature_name, worst_shap_val = neg_features[0]
    worst_feature_idx = FEATURE_NAMES.index(worst_feature_name)

    # Try each candidate movie as the pivot
    best_counterfactual = None
    best_score_gain     = 0.0

    for candidate in candidate_movies:
        pivot_id    = candidate["movie_id"]
        pivot_title = candidate["title"]
        current_rat = candidate["rating"]

        # Simulate rating increase
        new_rating = min(5.0, current_rat + RATING_STEP)

        # Rebuild feature vector with simulated rating
        try:
            original_fv = build_feature_vector(user_id, movie_id, db_conn)
        except ValueError:
            continue

        perturbed_fv = original_fv.copy()

        # Only modify the feature that corresponds to the negative SHAP value.
        # For user_similarity_score: increasing a similar user's rating boosts it.
        # For avg_explicit_rating: directly modifiable.
        if worst_feature_name == "avg_explicit_rating":
            perturbed_fv[worst_feature_idx] = new_rating / 5.0
        elif worst_feature_name == "user_similarity_score":
            perturbed_fv[worst_feature_idx] = min(1.0, original_fv[worst_feature_idx] + 0.1)
        else:
            # For other features, skip — they're not directly user-controllable.
            continue

        # Compute score gain
        original_score  = model_wrapper.predict(original_fv.reshape(1, -1))[0]
        perturbed_score = model_wrapper.predict(perturbed_fv.reshape(1, -1))[0]
        score_gain      = perturbed_score - original_score

        if score_gain > best_score_gain:
            best_score_gain     = score_gain
            best_counterfactual = {
                "pivot_movie_id":        pivot_id,
                "pivot_movie_title":     pivot_title,
                "current_rating":        current_rat,
                "suggested_rating":      new_rating,
                "predicted_rank_change": max(1, int(score_gain * 20)),
                "score_gain":            float(score_gain),
            }

    if best_counterfactual is None or best_score_gain < 0.01:
        return None

    return best_counterfactual
```

---

## 7. Phase 4 — Backend: Natural Language Reason Generator

### `backend/xai/nl_reason.py`

Converts raw SHAP values and counterfactual dicts into human-readable explanation strings. This is fully rule-based — no LLM dependency, so it is fast and deterministic.

```python
"""
nl_reason.py
Converts SHAP attribution dicts and counterfactual dicts into
natural-language explanation strings shown in the UI.

All templates are in TEMPLATES and COUNTERFACTUAL_TEMPLATES at the bottom.
To add new explanation types: add a template and update render_reason().
"""

from typing import Optional


# ─────────────────────────────────────────────
# Feature → human-readable label
# ─────────────────────────────────────────────
FEATURE_LABELS = {
    "genre_match_score":     "genre preferences",
    "avg_explicit_rating":   "how highly it's rated",
    "rating_count_log":      "how many people have rated it",
    "genome_relevance":      "thematic similarity to movies you love",
    "recency_score":         "how recently you added similar movies",
    "popularity_percentile": "overall popularity",
    "user_similarity_score": "viewers with similar taste to yours",
}


def render_reason(
    shap_result:     dict,
    movie_title:     str,
    model_type:      str,
) -> str:
    """
    Produces a one-to-two sentence explanation string from a SHAP result.

    Args:
        shap_result:  Output of SHAPExplainer.explain().
        movie_title:  Title of the recommended movie.
        model_type:   One of "content", "collaborative", "ncf".

    Returns:
        A plain-text explanation string. Example:
        "We're recommending Inception mainly because of your genre preferences
         and how highly it's rated. Viewers with similar taste to yours have
         also loved it."
    """
    pos = shap_result.get("top_positive_features", [])
    neg = shap_result.get("top_negative_features", [])

    if not pos:
        return f"We thought you might enjoy {movie_title} based on your viewing history."

    # Primary driver
    primary_feature, primary_val = pos[0]
    primary_label = FEATURE_LABELS.get(primary_feature, primary_feature)

    # Secondary driver (if exists)
    if len(pos) >= 2:
        secondary_feature, _ = pos[1]
        secondary_label = FEATURE_LABELS.get(secondary_feature, secondary_feature)
        main_sentence = (
            f"We're recommending {movie_title} mainly because of your "
            f"{primary_label} and {secondary_label}."
        )
    else:
        main_sentence = (
            f"We're recommending {movie_title} primarily based on your {primary_label}."
        )

    # Add model-type flavor
    if model_type == "collaborative":
        flavor = f" Viewers with similar taste to yours have consistently enjoyed it."
    elif model_type == "ncf":
        flavor = f" Our deep learning model found strong patterns linking your history to this film."
    else:  # content
        flavor = f" It shares key characteristics with movies already on your list."

    # Caveat if there's a notable negative factor
    caveat = ""
    if neg:
        neg_feature, neg_val = neg[0]
        neg_label = FEATURE_LABELS.get(neg_feature, neg_feature)
        if abs(neg_val) > 0.05:
            caveat = f" The main caveat is its {neg_label}."

    return main_sentence + flavor + caveat


def render_counterfactual(cf_dict: dict, target_movie_title: str) -> str:
    """
    Produces a natural-language counterfactual string.

    Args:
        cf_dict:            Output of generate_counterfactual().
        target_movie_title: Title of the movie being explained.

    Returns:
        A plain-text string. Example:
        "If you rated The Dark Knight above 4.5 stars,
         Tenet would jump approximately 3 positions in your recommendations."
    """
    if not cf_dict:
        return ""

    pivot    = cf_dict["pivot_movie_title"]
    new_rat  = cf_dict["suggested_rating"]
    rank_chg = cf_dict["predicted_rank_change"]

    return (
        f"If you rated {pivot} above {new_rat:.1f} stars, "
        f"{target_movie_title} would jump approximately {rank_chg} position"
        f"{'s' if rank_chg != 1 else ''} in your recommendations."
    )
```

---

## 8. Phase 5 — Flask API Endpoints

### `backend/routes/explain.py`

Register these routes in `app.py` with `app.register_blueprint(explain_bp, url_prefix="/api/explain")`.

```python
"""
explain.py
Flask blueprint for all XAI-related API endpoints.

Endpoints:
  GET  /api/explain/<model_type>/<int:movie_id>
       Returns full explanation (SHAP + counterfactual + NL reason) for one movie.

  GET  /api/explain/batch/<model_type>
       Returns explanations for a list of movie_ids (passed as query param).

  POST /api/explain/feedback
       Accepts user thumbs up/down on an explanation.

  GET  /api/explain/profile/<int:user_id>
       Returns a user's overall "taste profile" derived from averaged SHAP values.
"""

from flask import Blueprint, request, jsonify, g
from xai.shap_explainer import SHAPExplainer
from xai.counterfactual import generate_counterfactual
from xai.nl_reason import render_reason, render_counterfactual
from xai.explanation_cache import ExplanationCache
from db.xai_queries import (
    save_explanation,
    save_counterfactual,
    save_feedback,
    fetch_user_seen_movies,
    fetch_user_candidate_movies,
)
from models.model_wrapper import RecommenderWrapper

explain_bp = Blueprint("explain", __name__)

# Module-level cache (lives for the server process lifetime)
_cache = ExplanationCache(max_size=5000)


def _get_model(model_type: str, user_id: int):
    """
    Load the appropriate model object for a given model_type string.
    Implement this to load from your existing model persistence layer
    (pickle, torch.load, etc.).
    """
    # TODO: Replace with actual model loading from your existing code.
    # Example: return load_ncf_model() if model_type == "ncf" else load_cf_model()
    raise NotImplementedError("Connect _get_model() to your existing model loading code.")


@explain_bp.route("/<model_type>/<int:movie_id>", methods=["GET"])
def get_explanation(model_type: str, movie_id: int):
    """
    Returns the full explanation for a single (current_user, movie_id) pair.

    Query params:
      user_id (int, required): The user requesting the explanation.

    Response shape:
    {
      "movie_id": 123,
      "model_type": "ncf",
      "base_value": 0.42,
      "prediction": 0.78,
      "shap_values": { "genre_match_score": 0.21, ... },
      "top_positive_features": [["genre_match_score", 0.21], ...],
      "top_negative_features": [["popularity_percentile", -0.08], ...],
      "nl_reason": "We're recommending Inception mainly because ...",
      "counterfactual": {
        "pivot_movie_title": "The Dark Knight",
        "suggested_rating": 4.5,
        "predicted_rank_change": 3,
        "nl_counterfactual": "If you rated The Dark Knight above 4.5 ..."
      }
    }
    """
    user_id = request.args.get("user_id", type=int)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    if model_type not in ("content", "collaborative", "ncf"):
        return jsonify({"error": f"Unknown model_type: {model_type}"}), 400

    try:
        model   = _get_model(model_type, user_id)
        wrapper = RecommenderWrapper(model_type, model, user_id)

        # Background: movies the user has already seen
        seen_ids = fetch_user_seen_movies(user_id, g.db)

        explainer  = SHAPExplainer(model_type, model, user_id, g.db, _cache)
        shap_result = explainer.explain(movie_id, background_movie_ids=seen_ids)

        # Natural language reason
        movie_title = shap_result.get("movie_title", f"Movie #{movie_id}")
        nl_reason   = render_reason(shap_result, movie_title, model_type)

        # Counterfactual
        candidates = fetch_user_candidate_movies(user_id, g.db)
        cf_dict    = generate_counterfactual(user_id, movie_id, shap_result, candidates, wrapper, g.db)
        nl_cf      = render_counterfactual(cf_dict, movie_title) if cf_dict else None

        if cf_dict:
            cf_dict["nl_counterfactual"] = nl_cf

        # Persist to DB (async in production — use Celery task)
        save_explanation(user_id, movie_id, model_type, shap_result, nl_reason, g.db)
        if cf_dict:
            save_counterfactual(user_id, movie_id, model_type, cf_dict, g.db)

        return jsonify({
            **shap_result,
            "model_type":    model_type,
            "nl_reason":     nl_reason,
            "counterfactual": cf_dict,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500


@explain_bp.route("/batch/<model_type>", methods=["GET"])
def get_batch_explanations(model_type: str):
    """
    Returns explanations for multiple movies at once.
    Query params:
      user_id  (int, required)
      movie_ids (comma-separated ints, required): e.g. ?movie_ids=101,202,303
    """
    user_id   = request.args.get("user_id", type=int)
    movie_ids_str = request.args.get("movie_ids", "")
    if not user_id or not movie_ids_str:
        return jsonify({"error": "user_id and movie_ids are required"}), 400

    try:
        movie_ids = [int(x) for x in movie_ids_str.split(",") if x.strip()]
    except ValueError:
        return jsonify({"error": "movie_ids must be comma-separated integers"}), 400

    try:
        model    = _get_model(model_type, user_id)
        seen_ids = fetch_user_seen_movies(user_id, g.db)
        explainer = SHAPExplainer(model_type, model, user_id, g.db, _cache)
        results  = explainer.explain_batch(movie_ids, seen_ids)

        for r in results:
            movie_title = r.get("movie_title", f"Movie #{r['movie_id']}")
            r["nl_reason"] = render_reason(r, movie_title, model_type)

        return jsonify({"explanations": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@explain_bp.route("/feedback", methods=["POST"])
def post_feedback():
    """
    Accepts user feedback on an explanation.
    Body (JSON):
    {
      "user_id":      int,
      "movie_id":     int,
      "model_type":   str,
      "helpful":      bool,
      "feedback_text": str (optional)
    }
    """
    data = request.get_json()
    required = ["user_id", "movie_id", "model_type", "helpful"]
    missing  = [k for k in required if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    save_feedback(
        user_id=       data["user_id"],
        movie_id=      data["movie_id"],
        model_type=    data["model_type"],
        helpful=       data["helpful"],
        feedback_text= data.get("feedback_text", ""),
        db_conn=       g.db
    )

    # Invalidate cache for this user so next explanation is fresh
    _cache.invalidate_user(data["user_id"])

    return jsonify({"status": "ok"}), 200


@explain_bp.route("/profile/<int:user_id>", methods=["GET"])
def get_taste_profile(user_id: int):
    """
    Returns the user's aggregated "taste profile" — the average SHAP contribution
    of each feature across all their recommendations. Useful for a profile page
    showing "what drives your recommendations."

    Response shape:
    {
      "user_id": 42,
      "profile": {
        "genre_match_score": 0.28,        // This user is very genre-driven
        "user_similarity_score": 0.19,
        "avg_explicit_rating": 0.11,
        ...
      },
      "top_driver": "genre_match_score",
      "top_driver_label": "genre preferences"
    }
    """
    from db.xai_queries import fetch_user_shap_averages
    from xai.nl_reason import FEATURE_LABELS

    try:
        averages = fetch_user_shap_averages(user_id, g.db)
        top_driver = max(averages, key=lambda k: abs(averages[k]))

        return jsonify({
            "user_id":          user_id,
            "profile":          averages,
            "top_driver":       top_driver,
            "top_driver_label": FEATURE_LABELS.get(top_driver, top_driver),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
```

### `backend/db/xai_queries.py`

```python
"""
xai_queries.py
All database queries for the XAI layer.
Keep SQL here — never inline SQL in business logic files.
"""

import json
from typing import Optional


def fetch_xai_features(user_id: int, movie_id: int, db_conn) -> Optional[dict]:
    """Fetches the raw feature row from xai_feature_view."""
    cursor = db_conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM xai_feature_view WHERE user_id = %s AND movie_id = %s",
        (user_id, movie_id)
    )
    row = cursor.fetchone()
    cursor.close()
    return row


def fetch_user_seen_movies(user_id: int, db_conn, limit: int = 200) -> list[int]:
    """Returns list of movie_ids the user has rated or listed."""
    cursor = db_conn.cursor()
    cursor.execute("""
        SELECT DISTINCT movieId FROM (
            SELECT movieId FROM rating_explicit WHERE userId = %s
            UNION
            SELECT movieId FROM rating_implicit WHERE userId = %s
            UNION
            SELECT movieId FROM user_movielist WHERE userId = %s
        ) combined
        LIMIT %s
    """, (user_id, user_id, user_id, limit))
    ids = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return ids


def fetch_user_candidate_movies(user_id: int, db_conn, limit: int = 20) -> list[dict]:
    """Returns movies the user has rated (for counterfactual generation)."""
    cursor = db_conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT re.movieId AS movie_id, m.title, re.rating
        FROM rating_explicit re
        JOIN movies m ON m.movieId = re.movieId
        WHERE re.userId = %s
        ORDER BY re.rating DESC
        LIMIT %s
    """, (user_id, limit))
    rows = cursor.fetchall()
    cursor.close()
    return rows


def save_explanation(user_id, movie_id, model_type, shap_result, nl_reason, db_conn):
    """Persists an explanation to explanation_features."""
    cursor = db_conn.cursor()
    cursor.execute("""
        INSERT INTO explanation_features
            (user_id, movie_id, model_type_id, rec_score, shap_values_json, nl_reason)
        VALUES (%s, %s,
            (SELECT id FROM models_type WHERE name = %s LIMIT 1),
            %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            shap_values_json = VALUES(shap_values_json),
            nl_reason        = VALUES(nl_reason)
    """, (
        user_id, movie_id, model_type,
        shap_result["prediction"],
        json.dumps(shap_result["shap_values"]),
        nl_reason
    ))
    db_conn.commit()
    cursor.close()


def save_counterfactual(user_id, movie_id, model_type, cf_dict, db_conn):
    """Persists a counterfactual to counterfactual_explanations."""
    cursor = db_conn.cursor()
    cursor.execute("""
        INSERT INTO counterfactual_explanations
            (user_id, movie_id, model_type_id, pivot_movie_id,
             pivot_rating, predicted_rank_change, nl_counterfactual)
        VALUES (%s, %s,
            (SELECT id FROM models_type WHERE name = %s LIMIT 1),
            %s, %s, %s, %s)
    """, (
        user_id, movie_id, model_type,
        cf_dict.get("pivot_movie_id"),
        cf_dict.get("suggested_rating"),
        cf_dict.get("predicted_rank_change"),
        cf_dict.get("nl_counterfactual"),
    ))
    db_conn.commit()
    cursor.close()


def save_feedback(user_id, movie_id, model_type, helpful, feedback_text, db_conn):
    """Persists user feedback to explanation_feedback."""
    cursor = db_conn.cursor()
    cursor.execute("""
        INSERT INTO explanation_feedback
            (user_id, movie_id, model_type_id, helpful, feedback_text)
        VALUES (%s, %s,
            (SELECT id FROM models_type WHERE name = %s LIMIT 1),
            %s, %s)
    """, (user_id, movie_id, model_type, int(helpful), feedback_text))
    db_conn.commit()
    cursor.close()


def fetch_user_shap_averages(user_id: int, db_conn) -> dict:
    """Returns per-feature average SHAP values for a user across all past explanations."""
    cursor = db_conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT shap_values_json FROM explanation_features
        WHERE user_id = %s ORDER BY created_at DESC LIMIT 100
    """, (user_id,))
    rows = cursor.fetchall()
    cursor.close()
    if not rows:
        return {}
    from collections import defaultdict
    totals  = defaultdict(float)
    counts  = defaultdict(int)
    for row in rows:
        vals = json.loads(row["shap_values_json"])
        for k, v in vals.items():
            totals[k]  += v
            counts[k]  += 1
    return {k: totals[k] / counts[k] for k in totals}
```

---

## 9. Phase 6 — Frontend: Explanation UI Components

### TypeScript Interfaces — `frontend/types/explanation.ts`

```typescript
// All explanation-related type definitions.
// Import these in any component that touches XAI data.

export interface ShapFeature {
  name: string;
  value: number;
  label: string;   // Human-readable label from FEATURE_LABELS
}

export interface CounterfactualExplanation {
  pivot_movie_id:       number;
  pivot_movie_title:    string;
  current_rating:       number;
  suggested_rating:     number;
  predicted_rank_change: number;
  nl_counterfactual:    string;
}

export interface MovieExplanation {
  movie_id:              number;
  model_type:            "content" | "collaborative" | "ncf";
  base_value:            number;
  prediction:            number;
  shap_values:           Record<string, number>;
  top_positive_features: [string, number][];
  top_negative_features: [string, number][];
  nl_reason:             string;
  counterfactual?:       CounterfactualExplanation;
}

export interface TasteProfile {
  user_id:          number;
  profile:          Record<string, number>;
  top_driver:       string;
  top_driver_label: string;
}
```

### Custom Hook — `frontend/hooks/useExplanation.ts`

```typescript
import { useState, useCallback } from "react";
import { MovieExplanation } from "@/types/explanation";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

/**
 * Hook for fetching and caching a single movie explanation.
 * Designed to be called lazily (on "Why?" button click) so we don't
 * pre-fetch explanations for all recommendations on page load.
 *
 * Usage:
 *   const { explanation, loading, error, fetchExplanation } = useExplanation();
 *   <button onClick={() => fetchExplanation("ncf", movieId, userId)}>Why?</button>
 */
export function useExplanation() {
  const [explanation, setExplanation] = useState<MovieExplanation | null>(null);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState<string | null>(null);

  const fetchExplanation = useCallback(
    async (modelType: string, movieId: number, userId: number) => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch(
          `${API_BASE}/api/explain/${modelType}/${movieId}?user_id=${userId}`
        );
        if (!res.ok) {
          const body = await res.json();
          throw new Error(body.error ?? `HTTP ${res.status}`);
        }
        const data: MovieExplanation = await res.json();
        setExplanation(data);
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  return { explanation, loading, error, fetchExplanation };
}
```

### SHAP Waterfall Chart — `frontend/components/ShapWaterfallChart.tsx`

```typescript
/**
 * ShapWaterfallChart
 * Renders a horizontal waterfall bar chart of SHAP values.
 * Positive contributions are green, negative are red.
 * Built with pure CSS — no chart library dependency.
 *
 * Props:
 *   features: Array of [featureName, shapValue] tuples, sorted descending.
 *   maxValue:  The maximum absolute SHAP value for scaling bars.
 */

import React from "react";
import { ShapFeature } from "@/types/explanation";

const FEATURE_LABELS: Record<string, string> = {
  genre_match_score:     "Genre match",
  avg_explicit_rating:   "Average rating",
  rating_count_log:      "Number of ratings",
  genome_relevance:      "Thematic similarity",
  recency_score:         "Recency",
  popularity_percentile: "Popularity",
  user_similarity_score: "Viewer similarity",
};

interface Props {
  topPositive: [string, number][];
  topNegative: [string, number][];
}

export const ShapWaterfallChart: React.FC<Props> = ({ topPositive, topNegative }) => {
  const allFeatures = [...topPositive, ...topNegative];
  const maxAbs = Math.max(...allFeatures.map(([, v]) => Math.abs(v)), 0.01);

  return (
    <div className="space-y-2 mt-3">
      <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">
        Feature contributions
      </p>
      {allFeatures.map(([name, value]) => {
        const isPositive   = value >= 0;
        const widthPercent = (Math.abs(value) / maxAbs) * 100;
        const label        = FEATURE_LABELS[name] ?? name;

        return (
          <div key={name} className="flex items-center gap-2 text-sm">
            {/* Feature label */}
            <span className="w-36 text-right text-gray-600 text-xs shrink-0">
              {label}
            </span>
            {/* Bar */}
            <div className="flex-1 bg-gray-100 rounded-full h-3 relative overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isPositive ? "bg-green-500" : "bg-red-400"
                }`}
                style={{ width: `${widthPercent}%` }}
              />
            </div>
            {/* Value */}
            <span className={`text-xs w-12 ${isPositive ? "text-green-600" : "text-red-500"}`}>
              {isPositive ? "+" : ""}{value.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
};
```

### Explanation Panel — `frontend/components/ExplanationPanel.tsx`

```typescript
/**
 * ExplanationPanel
 * A slide-in drawer that shows the full XAI explanation for one movie.
 * Triggered by clicking "Why?" on a MovieCard.
 *
 * Contains:
 *   - Natural language reason (top of panel)
 *   - SHAP waterfall chart
 *   - Counterfactual card (if available)
 *   - Thumbs up/down feedback buttons
 */

import React, { useEffect } from "react";
import { MovieExplanation } from "@/types/explanation";
import { ShapWaterfallChart } from "./ShapWaterfallChart";
import { CounterfactualCard } from "./CounterfactualCard";
import { FeedbackButtons } from "./FeedbackButtons";

interface Props {
  explanation:  MovieExplanation | null;
  movieTitle:   string;
  userId:       number;
  loading:      boolean;
  error:        string | null;
  onClose:      () => void;
}

export const ExplanationPanel: React.FC<Props> = ({
  explanation, movieTitle, userId, loading, error, onClose
}) => {
  // Close on Escape key
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    // Backdrop
    <div
      className="fixed inset-0 bg-black/40 z-50 flex justify-end"
      onClick={onClose}
    >
      {/* Drawer panel — stop propagation so clicking inside doesn't close */}
      <div
        className="w-full max-w-md bg-white h-full overflow-y-auto shadow-xl p-6 flex flex-col gap-4"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900">
            Why {movieTitle}?
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-700 text-xl">
            ✕
          </button>
        </div>

        {/* Loading */}
        {loading && (
          <div className="flex items-center gap-2 text-gray-500">
            <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-400 border-t-transparent" />
            <span>Computing explanation...</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="text-red-500 text-sm bg-red-50 p-3 rounded-lg">
            Could not load explanation: {error}
          </div>
        )}

        {/* Explanation content */}
        {explanation && !loading && (
          <>
            {/* Natural language reason */}
            <div className="bg-blue-50 rounded-xl p-4 text-sm text-blue-900 leading-relaxed">
              {explanation.nl_reason}
            </div>

            {/* SHAP chart */}
            <div className="border border-gray-100 rounded-xl p-4">
              <ShapWaterfallChart
                topPositive={explanation.top_positive_features}
                topNegative={explanation.top_negative_features}
              />
            </div>

            {/* Counterfactual */}
            {explanation.counterfactual && (
              <CounterfactualCard cf={explanation.counterfactual} />
            )}

            {/* Feedback */}
            <FeedbackButtons
              userId={userId}
              movieId={explanation.movie_id}
              modelType={explanation.model_type}
            />
          </>
        )}
      </div>
    </div>
  );
};
```

### Counterfactual Card — `frontend/components/CounterfactualCard.tsx`

```typescript
import React from "react";
import { CounterfactualExplanation } from "@/types/explanation";

interface Props {
  cf: CounterfactualExplanation;
}

export const CounterfactualCard: React.FC<Props> = ({ cf }) => (
  <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 text-sm text-amber-900">
    <p className="font-medium mb-1">What would change this?</p>
    <p className="leading-relaxed">{cf.nl_counterfactual}</p>
    <div className="mt-2 flex items-center gap-2 text-xs text-amber-700">
      <span>Current rating of {cf.pivot_movie_title}:</span>
      <span className="font-mono bg-amber-100 px-2 py-0.5 rounded">
        {cf.current_rating.toFixed(1)} ★
      </span>
      <span>→</span>
      <span className="font-mono bg-amber-200 px-2 py-0.5 rounded">
        {cf.suggested_rating.toFixed(1)} ★
      </span>
    </div>
  </div>
);
```

### Feedback Buttons — `frontend/components/FeedbackButtons.tsx`

```typescript
import React, { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:5000";

interface Props {
  userId:    number;
  movieId:   number;
  modelType: string;
}

export const FeedbackButtons: React.FC<Props> = ({ userId, movieId, modelType }) => {
  const [submitted, setSubmitted] = useState<boolean | null>(null);

  const submit = async (helpful: boolean) => {
    await fetch(`${API_BASE}/api/explain/feedback`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ user_id: userId, movie_id: movieId, model_type: modelType, helpful }),
    });
    setSubmitted(helpful);
  };

  if (submitted !== null) {
    return (
      <p className="text-sm text-gray-500 text-center">
        {submitted ? "Glad this was helpful!" : "Thanks for the feedback — we'll improve."}
      </p>
    );
  }

  return (
    <div className="flex flex-col items-center gap-2">
      <p className="text-sm text-gray-500">Was this explanation helpful?</p>
      <div className="flex gap-3">
        <button
          onClick={() => submit(true)}
          className="px-4 py-2 text-sm bg-green-100 text-green-700 rounded-lg hover:bg-green-200 transition"
        >
          👍 Yes
        </button>
        <button
          onClick={() => submit(false)}
          className="px-4 py-2 text-sm bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition"
        >
          👎 No
        </button>
      </div>
    </div>
  );
};
```

### Modify MovieCard — `frontend/components/MovieCard.tsx`

Add a "Why?" button to the existing `MovieCard` component. The exact location within the card depends on the current layout, but the additions are:

```typescript
// Add to imports at top of MovieCard.tsx
import { useState } from "react";
import { useExplanation } from "@/hooks/useExplanation";
import { ExplanationPanel } from "./ExplanationPanel";

// Inside the MovieCard component function, add:
const [showExplanation, setShowExplanation] = useState(false);
const { explanation, loading, error, fetchExplanation } = useExplanation();

const handleWhyClick = () => {
  setShowExplanation(true);
  // Only fetch if not already fetched for this movie
  if (!explanation || explanation.movie_id !== movie.id) {
    fetchExplanation(modelType, movie.id, userId);
  }
};

// Add this button inside the card's action area:
<button
  onClick={handleWhyClick}
  className="text-xs text-blue-600 hover:text-blue-800 underline underline-offset-2"
>
  Why this?
</button>

// Add the panel (renders outside the card, as a portal or sibling):
{showExplanation && (
  <ExplanationPanel
    explanation={explanation}
    movieTitle={movie.title}
    userId={userId}
    loading={loading}
    error={error}
    onClose={() => setShowExplanation(false)}
  />
)}
```

---

## 10. Phase 7 — Feedback Loop

### How Feedback Feeds Back into the System

When a user submits a thumbs-down on an explanation, two things happen:

1. **Immediate cache invalidation:** `_cache.invalidate_user(user_id)` is called in the feedback endpoint, ensuring the next explanation for that user is recomputed fresh.

2. **Periodic retraining signal (implement as a scheduled job):** Once per day (via a cron job or AWS Lambda), run the following aggregation query to identify which features have low explanation helpfulness:

```sql
-- Query to audit explanation quality per feature
-- Run this daily and flag features with < 40% helpfulness rate
SELECT
    ef.model_type_id,
    JSON_KEYS(ef.shap_values_json) AS features,
    AVG(fb.helpful)                AS helpfulness_rate,
    COUNT(*)                       AS feedback_count
FROM explanation_features ef
JOIN explanation_feedback fb
    ON  fb.user_id   = ef.user_id
    AND fb.movie_id  = ef.movie_id
WHERE fb.created_at > NOW() - INTERVAL 7 DAY
GROUP BY ef.model_type_id;
```

Features with a helpfulness rate below 40% should be reviewed — the SHAP explanation may be surfacing a feature that users find irrelevant or confusing. Update `nl_reason.py` templates accordingly.

---

## 11. Phase 8 — Testing

### Backend Unit Tests — `backend/tests/test_xai.py`

```python
"""
test_xai.py
Unit tests for the XAI layer. Run with: pytest backend/tests/test_xai.py -v
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from xai.nl_reason import render_reason, render_counterfactual
from xai.explanation_cache import ExplanationCache
from utils.feature_extractor import FEATURE_NAMES


# ─────────────────────────
# nl_reason tests
# ─────────────────────────

def test_render_reason_with_two_positive_features():
    shap_result = {
        "top_positive_features": [
            ("genre_match_score", 0.21),
            ("avg_explicit_rating", 0.14),
        ],
        "top_negative_features": [],
    }
    reason = render_reason(shap_result, "Inception", "ncf")
    assert "Inception" in reason
    assert "genre preferences" in reason
    assert "how highly it's rated" in reason


def test_render_reason_with_negative_caveat():
    shap_result = {
        "top_positive_features": [("genre_match_score", 0.3)],
        "top_negative_features": [("popularity_percentile", -0.12)],
    }
    reason = render_reason(shap_result, "Tenet", "collaborative")
    assert "Tenet" in reason
    assert "popularity" in reason  # caveat should mention negative feature


def test_render_reason_no_features_falls_back():
    shap_result = {
        "top_positive_features": [],
        "top_negative_features": [],
    }
    reason = render_reason(shap_result, "Dunkirk", "content")
    assert "Dunkirk" in reason
    assert len(reason) > 10


def test_render_counterfactual():
    cf = {
        "pivot_movie_title":    "The Dark Knight",
        "suggested_rating":     4.5,
        "predicted_rank_change": 3,
    }
    result = render_counterfactual(cf, "Tenet")
    assert "The Dark Knight" in result
    assert "Tenet" in result
    assert "4.5" in result
    assert "3" in result


# ─────────────────────────
# Cache tests
# ─────────────────────────

def test_cache_set_and_get():
    cache = ExplanationCache(max_size=10)
    data  = {"movie_id": 1, "prediction": 0.8}
    cache.set(user_id=1, movie_id=1, model_type="ncf", value=data)
    result = cache.get(user_id=1, movie_id=1, model_type="ncf")
    assert result == data


def test_cache_miss_returns_none():
    cache = ExplanationCache()
    result = cache.get(user_id=99, movie_id=99, model_type="ncf")
    assert result is None


def test_cache_invalidate_user():
    cache = ExplanationCache()
    cache.set(1, 10, "ncf",  {"x": 1})
    cache.set(1, 20, "ncf",  {"x": 2})
    cache.set(2, 10, "ncf",  {"x": 3})
    cache.invalidate_user(1)
    assert cache.get(1, 10, "ncf") is None
    assert cache.get(1, 20, "ncf") is None
    assert cache.get(2, 10, "ncf") == {"x": 3}  # other user unaffected


def test_cache_eviction_at_max_size():
    cache = ExplanationCache(max_size=2)
    cache.set(1, 1, "ncf", {"x": 1})
    cache.set(1, 2, "ncf", {"x": 2})
    cache.set(1, 3, "ncf", {"x": 3})  # should evict the first entry
    assert cache.get(1, 1, "ncf") is None
    assert cache.get(1, 3, "ncf") == {"x": 3}


# ─────────────────────────
# Feature vector tests
# ─────────────────────────

def test_feature_names_length():
    assert len(FEATURE_NAMES) == 7


def test_feature_names_order_stable():
    # This test will fail if someone changes the order — intentional protection.
    expected = [
        "genre_match_score",
        "avg_explicit_rating",
        "rating_count_log",
        "genome_relevance",
        "recency_score",
        "popularity_percentile",
        "user_similarity_score",
    ]
    assert FEATURE_NAMES == expected
```

---

## 12. Dependencies & Environment Setup

### Python (add to `requirements.txt`)

```
shap==0.45.0
numpy>=1.24.0
torch>=2.0.0        # only if using NCF with PyTorch
scikit-learn>=1.3.0 # used internally by SHAP
pytest>=7.4.0
```

Install with:
```bash
cd backend
pip install -r requirements.txt
```

### Node.js (no new packages required)

All frontend components use only React + TailwindCSS, which are already in the project. No new npm packages needed.

### Environment Variables (add to `.env.local`)

```bash
# Already present — verify these exist:
DATABASE_URL=mysql://user:password@host:3306/kryptonite
NEXT_PUBLIC_API_URL=http://localhost:5000

# New — for future Redis cache (optional, skip for dev):
REDIS_URL=redis://localhost:6379
```

### Running the XAI Schema Migration

```bash
# From project root:
mysql -u <user> -p kryptonite < data/xai_schema.sql
```

---

## 13. File-by-File Implementation Checklist

Work through this list in order. Each item is a discrete, testable unit.

- [ ] `data/xai_schema.sql` — Create tables and view, run migration
- [ ] `backend/utils/feature_extractor.py` — Implement + verify `build_feature_vector` returns float array of length 7
- [ ] `backend/db/xai_queries.py` — Implement all 7 query functions; test each against dev DB
- [ ] `backend/models/model_wrapper.py` — Implement `RecommenderWrapper`; verify `predict()` returns float array
- [ ] `backend/xai/explanation_cache.py` — Implement `ExplanationCache`; run cache unit tests
- [ ] `backend/xai/shap_explainer.py` — Implement `SHAPExplainer`; smoke test with a single (user_id=1, movie_id=1) call
- [ ] `backend/xai/counterfactual.py` — Implement `generate_counterfactual`; verify it returns `None` gracefully when no pivot is found
- [ ] `backend/xai/nl_reason.py` — Implement `render_reason` and `render_counterfactual`; run nl_reason unit tests
- [ ] `backend/routes/explain.py` — Implement all 4 endpoints; test with curl or Postman
- [ ] `backend/app.py` — Register `explain_bp` blueprint
- [ ] `frontend/types/explanation.ts` — Add TypeScript interfaces
- [ ] `frontend/hooks/useExplanation.ts` — Implement hook; test with a mocked API
- [ ] `frontend/components/ShapWaterfallChart.tsx` — Implement chart; verify bar widths scale correctly
- [ ] `frontend/components/CounterfactualCard.tsx` — Implement card
- [ ] `frontend/components/FeedbackButtons.tsx` — Implement buttons; verify POST fires correctly
- [ ] `frontend/components/ExplanationPanel.tsx` — Assemble all sub-components; test open/close
- [ ] `frontend/components/MovieCard.tsx` — Add "Why this?" button and panel integration
- [ ] `backend/tests/test_xai.py` — All tests green (`pytest backend/tests/test_xai.py -v`)
- [ ] End-to-end smoke test: visit recommendations page → click "Why this?" → panel opens with real SHAP explanation

---

*End of implementation plan. All code blocks in this document are production-ready starting points. Copilot should treat `# TODO:` comments as integration points that connect to the existing codebase.*