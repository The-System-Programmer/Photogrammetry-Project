from Pipeline.Step1_Feature_extraction import features
from Pipeline.Step2_Feature_matching import matching

# Load feature and description
features = features()

# Load the matched features between pairs
matches = matching(features)

print("Features and matches has loaded")
