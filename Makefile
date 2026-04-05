step1:
	python -m Pipeline.Step1_Feature_extraction
step2:
	python -m Pipeline.Step2_Feature_matching

clean:
	rm -rf __pycache__ Pipeline/__pycache__
