.PHONY: Step1 Step2 run clean

Step1:
	python -m Pipeline.Step1_Feature_extraction
Step2:
	python -m Pipeline.Step2_Something

run: step1 step2

clean:
	rm -rf */__pycache__ */*/__pycache__
