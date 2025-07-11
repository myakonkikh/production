build:
	docker build -t myakonkikhnikita/production:latest .
	docker push myakonkikhnikita/production:latest

run:
	docker run --rm -p 5000:5000 --name production myakonkikhnikita/production:latest