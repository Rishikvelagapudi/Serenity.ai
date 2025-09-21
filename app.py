import streamlit as st
import sqlite3
import os
import pdfplumber
import docx2txt
import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# -------------------------
# Database Setup
# -------------------------
DB_FILE = "results.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        resume_name TEXT,
        job_title TEXT,
        hard_score REAL,
        semantic_score REAL,
        final_score REAL,
        missing_skills TEXT,
        feedback TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_result(resume_name, job_title, hard_score, semantic_score, final_score, missing_skills, feedback):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO results 
        (resume_name, job_title, hard_score, semantic_score, final_score, missing_skills, feedback)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (resume_name, job_title, hard_score, semantic_score, final_score, missing_skills, feedback))
    conn.commit()
    conn.close()

# -------------------------
# Text Extraction
# -------------------------
def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    text = ""
    if ext == ".pdf":
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except:
            file.seek(0)
            doc = fitz.open(stream=file.read(), filetype="pdf")
            for page in doc:
                text += page.get_text()
    elif ext in [".docx", ".doc"]:
        text = docx2txt.process(file)
    else:
        text = file.read().decode("utf-8", errors="ignore")
    return text.strip()

# -------------------------
# AI Scoring
# -------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and accurate

def hard_match(jd_skills, jd_qualifications, resume_text):
    resume_text_lower = resume_text.lower()
    must_have_matches = sum(skill.lower() in resume_text_lower for skill in jd_skills['must_have'])
    good_matches = sum(skill.lower() in resume_text_lower for skill in jd_skills['good_to_have'])
    qual_matches = sum(qual.lower() in resume_text_lower for qual in jd_qualifications)
    
    total_possible = len(jd_skills['must_have'])*0.5 + len(jd_skills['good_to_have'])*0.2 + len(jd_qualifications)*0.3
    if total_possible == 0:
        return 0
    score = (
        (must_have_matches * 0.5) +
        (good_matches * 0.2) +
        (qual_matches * 0.3)
    ) / total_possible * 100
    return round(score, 2)

def semantic_match(jd_text, resume_text):
    embeddings = model.encode([jd_text, resume_text], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity * 100, 2)

def final_score(hard, semantic, hard_weight=0.5, semantic_weight=0.5):
    return round(hard*hard_weight + semantic*semantic_weight,2)

def missing_skills(jd_skills, resume_text):
    resume_text_lower = resume_text.lower()
    missing_must = [s for s in jd_skills['must_have'] if s.lower() not in resume_text_lower]
    missing_good = [s for s in jd_skills['good_to_have'] if s.lower() not in resume_text_lower]
    return ", ".join(missing_must + missing_good) if missing_must + missing_good else "None"

def feedback(missing):
    if missing == "None":
        return "Excellent match. Minor improvements recommended."
    else:
        return f"Please consider adding missing skills/projects/certifications: {missing}"

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Resume Relevance Dashboard", layout="wide")
st.title("📊 Automated Resume Relevance Check System")

init_db()

# Sidebar: Upload JD & Resumes
st.sidebar.header("Upload Files")
jd_file = st.sidebar.file_uploader("Upload Job Description (JD)", type=["pdf","docx","txt"])
resume_files = st.sidebar.file_uploader("Upload Resumes", type=["pdf","docx"], accept_multiple_files=True)

# Filters
st.sidebar.header("Filters")
filter_job = st.sidebar.text_input("Filter by Job Title")
min_score = st.sidebar.slider("Minimum Final Score", 0, 100, 0)
search_term = st.sidebar.text_input("Search Resume or Feedback")

# -------------------------
# Process Uploads
# -------------------------
if jd_file and resume_files:
    jd_text = extract_text(jd_file)
    jd_skills = {
        'must_have': ['Python', 'Machine Learning', 'SQL'],
        'good_to_have': ['TensorFlow', 'PyTorch', 'Docker']
    }
    jd_qualifications = ['B.Tech', 'M.Tech']

    job_title = "Extracted Role"
    st.subheader(f"Job Description: {jd_file.name}")
    st.text_area("JD Content", jd_text, height=150)

    for resume in resume_files:
        try:
            resume_text = extract_text(resume)
            h_score = hard_match(jd_skills, jd_qualifications, resume_text)
            s_score = semantic_match(jd_text, resume_text)
            f_score = final_score(h_score, s_score)
            missing = missing_skills(jd_skills, resume_text)
            fb = feedback(missing)

            save_result(resume.name, job_title, h_score, s_score, f_score, missing, fb)
        except Exception as e:
            st.error(f"Error processing {resume.name}: {e}")

# -------------------------
# Display Results
# -------------------------
conn = sqlite3.connect(DB_FILE)
df = pd.read_sql_query("SELECT * FROM results", conn)
conn.close()

if not df.empty:
    # Apply filters
    if filter_job: df = df[df['job_title'].str.contains(filter_job, case=False, na=False)]
    if min_score > 0: df = df[df['final_score'] >= min_score]
    if search_term: df = df[df.apply(lambda row: search_term.lower() in row.to_string().lower(), axis=1)]

    # Highlight the best resume
    best_resume_row = df.loc[df['final_score'].idxmax()]
    st.subheader("🏆 Best Resume")
    st.markdown(f"**{best_resume_row['resume_name']}** with Final Score: {best_resume_row['final_score']}")
    st.markdown(f"**Missing Skills:** {best_resume_row['missing_skills']}")
    st.markdown(f"**Feedback:** {best_resume_row['feedback']}")

    st.subheader("📑 All Evaluation Results")
    st.dataframe(
        df[['resume_name','hard_score','semantic_score','final_score','missing_skills']],
        width='stretch'
    )

    st.subheader("📝 Feedback Details")
    for _, row in df.iterrows():
        with st.expander(f"Feedback: {row['resume_name']}"):
            st.write(f"**Job Title:** {row.get('job_title','N/A')}")
            st.write(f"**Final Score:** {row.get('final_score','N/A')}")
            st.write(f"**Missing Skills:** {row.get('missing_skills','N/A')}")
            st.write(f"**Feedback:** {row.get('feedback','N/A')}")
else:
    st.info("Upload JD and resumes to see results.")
