.PHONY: run #stop restart

default: run

run:
	@echo "Запуск Streamlit сервера..."
	streamlit run main.py

# stop:
# 	@echo "Остановка Streamlit сервера..."
# 	@pkill -f "streamlit run main.py" || true

# restart: stop run
# 	@echo "Сервер перезапущен"