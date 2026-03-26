.PHONY: test benchmark package

test:
	pip install pytest -q
	python -m pytest tests/ -v -x --tb=short

benchmark:
	bash run.sh

package:
	python scripts/build_release_archive.py
