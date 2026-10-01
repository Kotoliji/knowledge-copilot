.PHONY: install run test eval lint docker up down

install:
	pip install -r requirements.txt ruff

run:
	uvicorn app.main:app --reload --port 8000

test:
	pytest -q

eval:
	python -m evals.run_eval

lint:
	ruff check app tests evals

docker:
	docker build -t knowledge-copilot .

up:
	docker compose up --build -d

down:
	docker compose down
