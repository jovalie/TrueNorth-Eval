# Variables
VECTOR_STORE := vector_store/truenorth_kb_vectorstore
PYTHON := poetry run python

.PHONY: help build run run-interactive stop dev clean logs restart embed

help:
	@echo "TrueNorth Makefile Commands:"
	@echo "  build     - Build Docker containers"
	@echo "  run       - Start services in detached mode (builds vector store if needed)"
	@echo "  run-interactive - Start services in foreground (builds vector store if needed)"
	@echo "  stop      - Stop Docker services"
	@echo "  dev       - Stop, rebuild, and start services interactively"
	@echo "  embed     - Build/rebuild vector store from PDFs"
	@echo "  logs      - Follow Docker logs"
	@echo "  restart   - Restart Docker services"
	@echo "  clean     - Remove all Docker containers, images, and volumes"

build:
	@echo "🧹 Removing old images..."
	docker-compose down --rmi local 2>/dev/null || true
	@echo "🔨 Building new images..."
	docker-compose build

run: $(VECTOR_STORE)
	docker-compose up -d

$(VECTOR_STORE):
	@echo "📦 Vector store not found. Building it now..."
	$(PYTHON) knowledge.py

run-interactive: $(VECTOR_STORE)
	docker-compose up

stop:
	docker-compose down

embed:
	$(PYTHON) knowledge.py

dev: stop build run-interactive

logs:
	docker-compose logs -f

restart:
	docker-compose restart

clean:
	docker-compose down --rmi all --volumes
