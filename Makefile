.PHONY: all data features train ensemble eval report

all: report

data:
	python -m pipelines.build_data --configs configs/dataset.yaml

features:
	python -m pipelines.build_features +features=base

train:
	python -m pipelines.train +model=logreg +features=base +cv=cv

ensemble:
	python -m pipelines.ensemble +ensemble=convex

eval:
	python -m pipelines.evaluate +exp_id=EXP-0000

report:
	python -m pipelines.report
