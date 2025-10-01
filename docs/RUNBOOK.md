# RUNBOOK — FightIQ Definitive
## Daily Loop
- [ ] Journal start: `python -m tools.journal --note "..."`
- [ ] Adversarial check: `python -m pipelines.adv_validate`
- [ ] Train sweep or model(s): `python -m pipelines.train +model=logreg`
- [ ] Ensemble: `python -m pipelines.ensemble +ensemble=convex`
- [ ] Evaluate (with exp id): `python -m pipelines.evaluate +exp_id=EXP-XXXX`
- [ ] Paper figs: `python -m pipelines.paper_assets`
- [ ] Report: `python -m pipelines.report`
- [ ] Journal end; update tasks.
