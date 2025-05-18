APP   ?= main.py
PORT  ?= 8501
IMAGE ?= video-searching

.PHONY: default run build docker stop

default: run

run:
	@echo "Запуск Streamlit локально: $(APP)"
	streamlit run $(APP)

build:
	@echo "Сборка Docker-образа: $(IMAGE)"
	docker build -f infra/Dockerfile -t $(IMAGE) .

docker: build
	@echo "Запуск контейнера: http://localhost:$(PORT)"
	docker run --rm -p $(PORT):8501 --name $(IMAGE)-ctr $(IMAGE)
