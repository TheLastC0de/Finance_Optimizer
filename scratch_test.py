import requests
import json

# Test baseline against all 7 tasks (including new debt_avalanche)
print("Testing Strategic Baseline...")
r = requests.post("http://localhost:8000/baseline")
data = r.json()

print(f"BASELINE: {r.status_code}")
for x in data["results"]:
    print(f"  {x['task_id']}: score={x['score']}  resolved={x['resolved']}  steps={x['steps']}")
print(f"\nTOTAL SCORE: {data['total_score']}")
print(f"RESOLVED: {data['resolved']}")

# Check that global health score was blended in
# (Baseline scores used to be exactly 0.999, now they should be slightly different 
#  due to the 20% weight of the health score)
avg = data['average_score']
print(f"\nAverage Strategic Score: {avg}")
