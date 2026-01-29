all: 
	docker compose up -d bike-api node-exporter prometheus grafana 

stop: 
	docker compose down

evaluation:
	docker compose up -d evaluation

traffic-generation:
	docker compose up -d traffic-generation

fire-alert:
	docker compose up -d trigger-rmse
