# FightIQ — Definitive MMA Prediction Study (V3)

Leak-safe time CV with embargo, feature groups, OOF predictions, MLflow tracking, adversarial validation,
drift metrics, convex ensembling scaffolds, experiments registry, journal/tasks, and paper assets generator.
Ships with synthetic data so you can run end-to-end immediately.

## Quickstart
```bash
conda env create -f environment.yml
conda activate fightiq

# Build synthetic data & features
python -m pipelines.build_data cutoff=2024-12-31
python -m pipelines.build_features +features=base features.I_odds=false

# Train a couple of base models
python -m pipelines.train +model=logreg +features=base +cv=cv
python -m pipelines.train +model=rf +features=base +cv=cv

# Ensemble (random-search on simplex)
python -m pipelines.ensemble +ensemble=convex

# Adversarial validation & drift
python -m pipelines.adv_validate +cv=cv

# Evaluate (logs to MLflow) & generate paper figures
python -m pipelines.evaluate +exp_id=EXP-0001 +note="baseline evaluation"
python -m pipelines.paper_assets

# Generate a Markdown run report
python -m pipelines.report
```
