.PHONY: doc

test:
	python -m pytest -v
	
coverage:
	coverage run -m pytest -v; \
	coverage report -m --skip-empty

format:
	black --preview elemental_agents/ tests/ examples/; \
	isort elemental_agents/ tests/ examples/
	
clean:
	rm -r *.db htmlcov .coverage

lint:
	pylint --output-format=colorized elemental_agents/*

doc:
	pdoc --output-dir=./doc/html --footer-text="Elemental 0.1.0" elemental_agents
	
mypy:
	mypy elemental_agents/

build:
	poetry build
