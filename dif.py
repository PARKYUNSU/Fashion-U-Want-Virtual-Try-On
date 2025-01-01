import json
from deepdiff import DeepDiff

# JSON 파일 읽기
with open('00001_00_keypoints.json', 'r') as gen_file:
    generated_json = json.load(gen_file)

with open('original_00001_00_keypoints.json', 'r') as orig_file:
    original_json = json.load(orig_file)

# JSON 비교
diff = DeepDiff(original_json, generated_json, ignore_order=True)

# 차이점 출력
if diff:
    print("Differences found:")
    print(json.dumps(diff, indent=4))
else:
    print("The JSON files are identical.")