import json
import os
import subprocess

# Define the candidate profile you want to test
# You can edit these values to see how the ranker responds!
custom_candidate = {
    "candidate_id": "TEST_CANDIDATE_USER",
    "profile": {
        "current_title": "Senior Machine Learning Engineer",
        "years_of_experience": 7.0,
        "location": "Pune",
        "current_company": "Zomato AI"
    },
    "skills": [
        {"name": "Sentence Transformers", "proficiency": "expert", "duration_months": 36, "endorsements": 90},
        {"name": "Embeddings", "proficiency": "expert", "duration_months": 36, "endorsements": 85},
        {"name": "Qdrant", "proficiency": "expert", "duration_months": 24, "endorsements": 70},
        {"name": "FAISS", "proficiency": "expert", "duration_months": 24, "endorsements": 60},
        {"name": "Information Retrieval", "proficiency": "expert", "duration_months": 30, "endorsements": 80},
        {"name": "Vector Search", "proficiency": "expert", "duration_months": 24, "endorsements": 50},
        {"name": "Python", "proficiency": "expert", "duration_months": 80, "endorsements": 95}
    ],
    "redrob_signals": {
        "recruiter_response_rate": 0.85,
        "open_to_work_flag": True,
        "notice_period_days": 15,
        "github_score": 75,
        "connection_count": 520,
        "linkedin_connected": True,
        "search_appearance_score": 0.90,
        "interview_completion_rate": 0.95,
        "last_active_date": "2026-06-25",
        "skill_assessment_scores": {"Sentence Transformers": 95, "FAISS": 90}
    },
    "education": [
        {"degree": "M.Tech", "tier": "tier_1", "field_of_study": "Computer Science"}
    ],
    "certifications": [
        {"name": "AWS Certified Machine Learning", "year": 2024}
    ],
    "career_history": [
        {
            "description": "Led the development of semantic search systems utilizing Sentence Transformers and Qdrant. Deployed ranking models to production and conducted A/B testing."
        }
    ]
}

def main():
    db_path = "candidates.jsonl"
    
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found in this directory.")
        return

    print("Step 1: Adding custom candidate to candidates.jsonl...")
    # Append the custom candidate to the file
    with open(db_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(custom_candidate) + "\n")
        
    try:
        print("Step 2: Running rank.py on the updated dataset...")
        # Run rank.py
        subprocess.run(["python", "rank.py", "candidates.jsonl", "submission.csv"], check=True)
        
        print("\nStep 3: Checking results in submission.csv...")
        found = False
        with open("submission.csv", "r", encoding="utf-8") as f:
            for line in f:
                if "TEST_CANDIDATE_USER" in line:
                    parts = line.strip().split(",", 3)
                    print("\n" + "="*50)
                    print("CANDIDATE FOUND IN TOP 100!")
                    print(f"Candidate ID: {parts[0]}")
                    print(f"Rank:         #{int(parts[1]) + 1}")
                    print(f"Score:        {parts[2]}")
                    print(f"Reasoning:    {parts[3]}")
                    print("="*50 + "\n")
                    found = True
                    break
        
        if not found:
            print("\nCandidate was not ranked in the Top 100 (score was too low to qualify).")
            print("Try adjusting skills, title, or experience in test_resume.py to boost their score!\n")
            
    finally:
        # Clean up the candidates.jsonl file so we don't pollute the database
        print("Step 4: Cleaning up candidates.jsonl...")
        with open(db_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Rewrite everything except our test candidate
        with open(db_path, "w", encoding="utf-8") as f:
            for line in lines:
                if '"candidate_id": "TEST_CANDIDATE_USER"' not in line:
                    f.write(line)
        print("Cleanup complete!")

if __name__ == "__main__":
    main()
