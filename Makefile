make:
	make clean
	rm -f NNPraktikum.zip
	zip -r NNPraktikum.zip . --exclude Makefile -x *.git*

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete