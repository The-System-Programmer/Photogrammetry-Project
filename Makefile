.PHONY: Step1 Step2 Step3 run clean

Step1:
	python -m Pipeline.Step1_Feature_extraction
Step2:
	python -m Pipeline.Step2_Feature_matching
Step3:
	python -m Pipeline.Step3_Camera_Geometry

run: step1 step2

clean:
	rm -rf */__pycache__ */*/__pycache__
