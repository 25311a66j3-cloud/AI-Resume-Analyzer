import re

import streamlit as st
from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Skills that our first version knows how to identify.
SKILLS = [
    "python", "java", "c++", "javascript", "html", "css",
    "sql", "mysql", "postgresql", "mongodb",
    "excel", "power bi", "tableau",
    "machine learning", "deep learning", "data analysis",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "flask", "django", "streamlit", "react",
    "git", "github", "aws", "docker",
    "communication", "problem solving", "teamwork",
]

# These are example role suggestions, not live job vacancies.
JOB_ROLES = {
    "Python Developer": {"python", "sql", "git", "flask", "django"},
    "Data Analyst": {"python", "sql", "excel", "power bi", "pandas"},
    "Machine Learning Intern": {
        "python", "machine learning", "pandas", "numpy", "scikit-learn"
    },
    "Web Developer": {"html", "css", "javascript", "react", "git"},
    "Java Developer": {"java", "sql", "git", "mysql"},
}


def read_resume(uploaded_file):
    """Extract text from a PDF or DOCX file."""
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if uploaded_file.name.lower().endswith(".docx"):
        document = Document(uploaded_file)

        paragraphs = [paragraph.text for paragraph in document.paragraphs]

        # Also read text inside tables, which some resumes use.
        table_cells = [
            cell.text
            for table in document.tables
            for row in table.rows
            for cell in row.cells
        ]

        return "\n".join(paragraphs + table_cells)

    return ""


def find_skills(text):
    """Return known skills whose full names appear in the text."""
    text = text.lower()
    found = set()

    for skill in SKILLS:
        pattern = rf"(?<!\w){re.escape(skill)}(?!\w)"

        if re.search(pattern, text):
            found.add(skill)

    return found


def get_text_similarity(resume_text, job_text):
    """Compare the wording of the resume and job description."""
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, job_text])

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return float(similarity)


def get_role_suggestions(resume_skills):
    """Rank example roles by overlap with the resume's skills."""
    ranked_roles = []

    for role, role_skills in JOB_ROLES.items():
        matched = resume_skills & role_skills
        percentage = round(len(matched) / len(role_skills) * 100)
        ranked_roles.append((role, percentage, matched))

    return sorted(ranked_roles, key=lambda item: item[1], reverse=True)[:3]


st.set_page_config(
    page_title="AI Resume Analyzer and Job Matcher",
    page_icon="📄",
    layout="wide",
)

st.title("📄 AI Resume Analyzer and Job Matcher")
st.write(
    "Upload a resume and paste a job description to see how their "
    "skills and wording compare."
)

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx"],
)

job_description = st.text_area(
    "Paste the job description",
    height=220,
    placeholder="Example: We need a Python developer with SQL, Git and Flask...",
)

if st.button("Analyze Resume", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a PDF or DOCX resume.")

    elif not job_description.strip():
        st.warning("Please paste a job description.")

    else:
        try:
            resume_text = read_resume(uploaded_file)
        except Exception as error:
            st.error(f"Could not read the file: {error}")
            st.stop()

        if not resume_text.strip():
            st.error(
                "No readable text was found. If this is a scanned PDF, "
                "try a text-based PDF or a DOCX resume."
            )
            st.stop()

        resume_skills = find_skills(resume_text)
        job_skills = find_skills(job_description)

        matched_skills = sorted(resume_skills & job_skills)
        missing_skills = sorted(job_skills - resume_skills)

        text_similarity = get_text_similarity(
            resume_text,
            job_description,
        )

        if job_skills:
            skill_match = len(matched_skills) / len(job_skills)
            match_score = round(
                (skill_match * 0.7 + text_similarity * 0.3) * 100
            )
        else:
            match_score = round(text_similarity * 100)

        st.subheader("Analysis Results")

        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated match", f"{match_score}%")
        col2.metric("Matched skills", len(matched_skills))
        col3.metric("Skills to review", len(missing_skills))

        left, right = st.columns(2)

        with left:
            st.subheader("✅ Matched skills")
            if matched_skills:
                st.write(", ".join(matched_skills))
            else:
                st.write("No skills from our list matched.")

        with right:
            st.subheader("📚 Skills mentioned in the job")
            if missing_skills:
                st.write(", ".join(missing_skills))
            else:
                st.write("No missing skills from our list were found.")

        st.subheader("💡 Resume suggestions")

        if missing_skills:
            st.write(
                "Check whether you already have experience with any "
                "of these skills: " + ", ".join(missing_skills) + "."
            )
            st.write(
                "If you do, describe that experience clearly in your "
                "resume. Only add skills you actually have."
            )
        else:
            st.write(
                "Your resume mentions the recognized skills in this "
                "job description. Check that your projects show how "
                "you used them."
            )

        st.subheader("🎯 Possible roles based on your resume")
        for role, percentage, matched in get_role_suggestions(resume_skills):
            skills_text = ", ".join(sorted(matched)) if matched else "none yet"
            st.write(
                f"**{role}** — {percentage}% of this role's example "
                f"skills found. Matched: {skills_text}."
            )

        with st.expander("View extracted resume text"):
            st.text(resume_text[:10000])

        st.caption(
            "Scores are approximate. This app compares text and a small "
            "skills list; it cannot judge a person's full ability."
        )