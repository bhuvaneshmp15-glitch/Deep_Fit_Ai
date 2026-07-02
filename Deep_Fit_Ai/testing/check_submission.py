import csv
with open('submission.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print(f'Total rows: {len(rows)}')
print()
print('TOP 10:')
for r in rows[:10]:
    print(f"  #{r['rank']:>3}  {r['candidate_id']}  score={r['score']}  {r['reasoning'][:90]}")
print()
print('BOTTOM 5:')
for r in rows[-5:]:
    print(f"  #{r['rank']:>3}  {r['candidate_id']}  score={r['score']}  {r['reasoning'][:90]}")
