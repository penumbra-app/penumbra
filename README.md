# Penumbra

Penumbra (https://www.penumbra.mov) is a personalized movie recommendation system designed to learn what individual users enjoy and return ranked movie recommendations tailored to their taste.

The project is built around three cooperating recommendation components:

- Content-based recommendation
- Collaborative filtering
- Hybrid / reranking

The goal is not simply to combine as many models as possible. Penumbra keeps model changes only when they are understandable, testable, and supported by held-out evaluation.

---

# Recommendation System

## Content-Based Recommendation

The content system models a user's preferences from properties of movies they have previously rated.

It builds a reusable user taste profile and predicts preference for unseen movies using metadata such as:

- Genre
- Director
- Runtime
- Release era
- Language
- User rating history

The content model also produces confidence and reason signals so predictions can reflect how much evidence actually supports them.

---

## Collaborative Filtering

The collaborative system learns from rating behavior across users.

Its primary model is biased matrix factorization, which learns:

- Global rating behavior
- User biases
- Movie biases
- Latent user factors
- Latent movie factors

These learned representations allow the system to estimate how strongly a user may prefer an unseen movie based on population-level behavior.

The collaborative model includes:

- confidence-aware predictions
- explicit sparse-user handling
- explicit unknown-user and unknown-movie behavior
- batch prediction
- model persistence and loading

---

## Hybrid / Reranking

The hybrid layer determines the final ordering of candidate movies.

Penumbra supports multiple ranking configurations using signals such as:

- Personal preference
- Content-model prediction
- Collaborative prediction
- Movie quality
- Movie popularity

The strongest currently validated ranking model uses:

`Personal + Quality + Popularity → Pairwise Logistic Regression`

Week 3 also integrated real content and collaborative predictions into experimental hybrid rerankers.

Those experiments successfully produced a full multi-model architecture, but they did not outperform the simpler existing learned reranker on held-out evaluation.

Those experiments successfully produced an integrated multi-model architecture, but they did not outperform the simpler existing learned reranker on held-out evaluation.

Detailed ranking methodology, experiments, ablations, and benchmark history are documented in:

`docs/hybrid-reranking.md`

---

# Final Recommendation Pipeline

The current single-user Penumbra recommendation flow is:

```text
User Ratings
    ↓
Generate unseen movie candidates
    ↓
Personal preference + movie quality + popularity
    ↓
Pairwise logistic-regression reranker
    ↓
Rank candidates by learned relevance
    ↓
Retain larger high-relevance candidate pool
    ↓
Genre-based diversity reranking
    ↓
Final Top-K recommendations