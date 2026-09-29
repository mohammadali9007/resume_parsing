import re
import os
from io import BytesIO

import pandas as pd
import streamlit as st
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

st.set_page_config(page_title="ResumeIQ | Resume Parser", page_icon="📄", layout="wide")

SKILLS = [
    "python", "java", "c++", "javascript", "html", "css", "sql", "mysql",
    "postgresql", "mongodb", "django", "flask", "machine learning",
    "deep learning", "artificial intelligence", "natural language processing",
    "nlp", "computer vision", "tensorflow", "pytorch", "scikit-learn",
    "pandas", "numpy", "opencv", "transformers", "llm", "git", "github",
    "docker", "data analysis", "power bi", "excel"
]

def extract_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    if name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    return ""

def extract_email(text):
    m = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return m.group(0) if m else "Not Found"

def extract_phone(text):
    m = re.search(r"(?:\+?880[-\s]?)?01[3-9]\d{8}", text)
    return m.group(0) if m else "Not Found"

def extract_cgpa(text):
    m = re.search(r"(?:cgpa|gpa)\s*[:\-]?\s*(\d(?:\.\d{1,2})?)", text, re.I)
    return m.group(1) if m else "Not Found"

def extract_name(text):
    for line in (x.strip() for x in text.splitlines() if x.strip()):
        if 2 <= len(line.split()) <= 5 and len(line) < 60 and "@" not in line and not any(c.isdigit() for c in line):
            if not any(k in line.lower() for k in ["resume", "curriculum vitae", "profile", "education", "contact"]):
                return line
    return "Not Found"

def extract_section(text, section_names):
    lines = text.splitlines()
    collecting, result = False, []
    headings = ["education", "experience", "skills", "projects", "certification", "certifications", "summary", "objective", "work history"]
    for line in lines:
        clean = line.strip()
        lower = clean.lower().strip(" :\t")
        if any(lower == s or lower.startswith(s + " ") or lower.startswith(s + ":") for s in section_names):
            collecting = True
            continue
        if collecting:
            if any(lower == h or lower.startswith(h + " ") or lower.startswith(h + ":") for h in headings):
                break
            if clean:
                result.append(clean)
    return "\n".join(result[:12])

def parse_resume(text):
    return {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Phone": extract_phone(text),
        "CGPA": extract_cgpa(text),
        "Education": extract_section(text, ["education", "academic background", "qualification"]),
        "Experience": extract_section(text, ["experience", "work experience", "employment"]),
        "Skills": extract_section(text, ["skills", "technical skills", "core skills"]),
        "Projects": extract_section(text, ["projects", "academic projects", "personal projects"]),
        "Certifications": extract_section(text, ["certifications", "certificates"]),
    }

def find_skills(text):
    text = text.lower()
    found = []
    for skill in SKILLS:
        pattern = r"(?<![a-z0-9+#])" + re.escape(skill) + r"(?![a-z0-9+#])"
        if re.search(pattern, text):
            found.append(skill)
    return sorted(set(found))

def make_pdf(filename, data, overall, skill_score, similarity, matched, missing):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    styles["Title"].alignment = TA_CENTER
    story = [Paragraph("Resume Analysis Report", styles["Title"]), Spacer(1, 14),
             Paragraph(f"<b>Resume:</b> {filename}", styles["Normal"]), Spacer(1, 12),
             Paragraph("Candidate Information", styles["Heading2"])]
    rows = [["Name", str(data.get("Name", "Not Found"))], ["Email", str(data.get("Email", "Not Found"))],
            ["Phone", str(data.get("Phone", "Not Found"))], ["CGPA", str(data.get("CGPA", "Not Found"))]]
    table = Table(rows, colWidths=[110, 390])
    table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("BACKGROUND",(0,0),(0,-1),colors.lightgrey),("VALIGN",(0,0),(-1,-1),"TOP")]))
    story += [table, Spacer(1, 14), Paragraph("Job Matching Score", styles["Heading2"])]
    score_table = Table([["Overall Match", f"{overall:.2f}%"], ["Skill Match", f"{skill_score:.2f}%"], ["Text Similarity", f"{similarity:.2f}%"]], colWidths=[160, 340])
    score_table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("BACKGROUND",(0,0),(0,-1),colors.lightgrey)]))
    story += [score_table, Spacer(1, 12)]
    for title, value in [("Matching Skills", ", ".join(matched) or "None"), ("Missing Skills", ", ".join(missing) or "None")]:
        story += [Paragraph(title, styles["Heading2"]), Paragraph(value, styles["Normal"]), Spacer(1, 10)]
    for title in ["Education", "Experience", "Skills", "Projects", "Certifications"]:
        story += [Paragraph(title, styles["Heading2"]), Paragraph(str(data.get(title) or "Not Found").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>"), styles["Normal"]), Spacer(1, 9)]
    story += [Spacer(1, 8), Paragraph("<b>Note:</b> Score is based on TF-IDF text similarity and predefined skill matching. It is not an automatic hiring decision.", styles["Normal"])]
    doc.build(story)
    buffer.seek(0)
    return buffer

# Styling
st.markdown("""
<style>
.block-container{max-width:1400px;padding-top:2rem}
.hero{background:linear-gradient(135deg,#4f46e5,#8b5cf6);padding:30px;border-radius:20px;color:white;margin-bottom:24px}
.hero h1{font-size:40px;margin:0}
.kpi{border:1px solid #334155;border-radius:15px;padding:18px;text-align:center;background:rgba(100,116,139,.08)}
.kpi-label{font-size:12px;opacity:.75;text-transform:uppercase}
.kpi-value{font-size:28px;font-weight:800}
</style>
""", unsafe_allow_html=True)

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.parsed = {}
    st.session_state.texts = {}
    st.session_state.job_skills = []
    st.session_state.job_description = ""

with st.sidebar:
    st.title("📄 ResumeIQ")
    st.caption("Resume Parser & Job Matcher")
    page = st.radio("Navigation", ["🏠 Dashboard", "📤 Analyze Resumes", "🏆 Candidates", "📊 Analytics", "📋 Parsed Data", "ℹ️ About"])

if page == "🏠 Dashboard":
    st.markdown('<div class="hero"><h1>ResumeIQ</h1><p>Resume Parsing & Intelligent Job Matching</p><p>Upload multiple CVs, extract candidate details, and compare resumes with a job description.</p></div>', unsafe_allow_html=True)
    df = st.session_state.results
    if df is None:
        st.info("Start from **Analyze Resumes** to upload CVs and compare them with a job description.")
    else:
        cols = st.columns(4)
        metrics = [("Resumes", len(df)), ("Average Match", f"{df['Overall Match (%)'].mean():.1f}%"), ("Highest Match", f"{df['Overall Match (%)'].max():.1f}%"), ("Required Skills", len(st.session_state.job_skills))]
        for col, (label, value) in zip(cols, metrics):
            col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)
        st.subheader("Overall Match by Resume")
        st.bar_chart(df.set_index("Resume")[["Overall Match (%)"]])

elif page == "📤 Analyze Resumes":
    st.title("📤 Analyze Resumes")
    files = st.file_uploader("Upload one or multiple resumes (PDF, DOCX, TXT)", type=["pdf","docx","txt"], accept_multiple_files=True)
    job = st.text_area("Paste the job description", height=220, placeholder="Example: Looking for a Python/ML Intern with Python, Machine Learning, NLP, Pandas and Scikit-learn skills.")
    if st.button("🚀 Analyze Resumes", type="primary", use_container_width=True):
        if not files:
            st.error("Please upload at least one resume.")
            st.stop()
        if not job.strip():
            st.error("Please enter a job description.")
            st.stop()
        texts, parsed = {}, {}
        progress = st.progress(0)
        for i, file in enumerate(files):
            try:
                content = extract_text(file)
                if not content.strip():
                    st.warning(f"No readable text found in {file.name}. Scanned PDFs may need OCR.")
                    continue
                texts[file.name] = content
                parsed[file.name] = parse_resume(content)
            except Exception as e:
                st.warning(f"Could not process {file.name}: {e}")
            progress.progress((i + 1) / len(files))
        if not texts:
            st.error("No resume could be processed.")
            st.stop()
        docs = [job] + [texts[n] for n in texts]
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1,2)).fit_transform(docs)
        sims = cosine_similarity(matrix[0], matrix[1:])[0]
        job_skills = find_skills(job)
        rows = []
        for i, filename in enumerate(texts):
            resume_skills = find_skills(texts[filename])
            matched = [s for s in job_skills if s in resume_skills]
            missing = [s for s in job_skills if s not in resume_skills]
            skill_score = (len(matched) / len(job_skills) * 100) if job_skills else 0
            similarity = float(sims[i] * 100)
            overall = similarity * .60 + skill_score * .40
            rows.append({"Resume": filename, "Overall Match (%)": round(overall,2), "Text Similarity (%)": round(similarity,2), "Skill Match (%)": round(skill_score,2), "Matching Skills": ", ".join(matched) or "None", "Missing Skills": ", ".join(missing) or "None"})
        df = pd.DataFrame(rows).sort_values("Overall Match (%)", ascending=False).reset_index(drop=True)
        st.session_state.results = df
        st.session_state.parsed = parsed
        st.session_state.texts = texts
        st.session_state.job_skills = job_skills
        st.session_state.job_description = job
        st.success(f"Analyzed {len(df)} resume(s) successfully.")
        st.dataframe(df, use_container_width=True)

elif page == "🏆 Candidates":
    df = st.session_state.results
    if df is None:
        st.info("Analyze resumes first.")
    else:
        st.title("🏆 Candidate Comparison")
        for idx, row in df.iterrows():
            filename = row["Resume"]
            data = st.session_state.parsed[filename]
            with st.expander(f"Rank {idx+1} | {filename} | {row['Overall Match (%)']}% match", expanded=False):
                c1, c2 = st.columns([1,2])
                c1.metric("Overall Match", f"{row['Overall Match (%)']}%")
                c2.write(f"**{data['Name']}**  \nEmail: {data['Email']}  \nPhone: {data['Phone']}  \nCGPA: {data['CGPA']}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Overall", f"{row['Overall Match (%)']}%")
                m2.metric("Skill Match", f"{row['Skill Match (%)']}%")
                m3.metric("Text Similarity", f"{row['Text Similarity (%)']}%")
                st.markdown("**✅ Matching Skills:** " + (row["Matching Skills"] or "None"))
                st.markdown("**❌ Missing Skills:** " + (row["Missing Skills"] or "None"))
                for section in ["Education","Experience","Skills","Projects","Certifications"]:
                    st.markdown(f"**{section}**")
                    st.write(data.get(section) or "Not Found")
                pdf = make_pdf(filename, data, row["Overall Match (%)"], row["Skill Match (%)"], row["Text Similarity (%)"], row["Matching Skills"].split(", ") if row["Matching Skills"] != "None" else [], row["Missing Skills"].split(", ") if row["Missing Skills"] != "None" else [])
                safe = re.sub(r"[^A-Za-z0-9_-]", "_", os.path.splitext(filename)[0])
                st.download_button("📄 Download Candidate PDF Report", pdf, file_name=f"{safe}_report.pdf", mime="application/pdf", key=f"pdf_{idx}")

elif page == "📊 Analytics":
    df = st.session_state.results
    if df is None:
        st.info("Analyze resumes first.")
    else:
        st.title("📊 Analytics")
        st.subheader("Overall Match")
        st.bar_chart(df.set_index("Resume")[["Overall Match (%)"]])
        st.subheader("Skill Match vs Text Similarity")
        st.bar_chart(df.set_index("Resume")[["Skill Match (%)", "Text Similarity (%)"]])
        st.subheader("Required Skills")
        st.write(", ".join(st.session_state.job_skills) or "No predefined skills detected.")
        st.download_button("⬇️ Download Results CSV", df.to_csv(index=False).encode("utf-8"), file_name="resume_job_matching_results.csv", mime="text/csv")

elif page == "📋 Parsed Data":
    if not st.session_state.parsed:
        st.info("Analyze resumes first.")
    else:
        st.title("📋 Parsed Resume Data")
        st.dataframe(pd.DataFrame(st.session_state.parsed).T, use_container_width=True)
        parsed_csv = pd.DataFrame(st.session_state.parsed).T.to_csv(index=True).encode("utf-8")
        st.download_button("Download Parsed Data CSV", parsed_csv, file_name="parsed_resumes.csv", mime="text/csv")

elif page == "ℹ️ About":
    st.title("ℹ️ About ResumeIQ")
    st.markdown("""
    **ResumeIQ** is a resume parsing and job matching web application built with Python, Streamlit, and NLP techniques.

    **Features**
    - Upload multiple PDF, DOCX, and TXT resumes
    - Extract contact information and resume sections
    - Detect predefined skills
    - Compare resumes with a job description using TF-IDF and cosine similarity
    - Calculate a combined match score (60% text similarity + 40% skill match)
    - Compare candidates and export CSV/PDF reports

    **Important:** This is a basic algorithmic matching tool, not an automatic hiring decision. Results should be reviewed by a human. Scanned image-only PDFs may require OCR.
    """)

st.markdown("---")
st.caption("ResumeIQ • Resume Parser & Job Matching System • Python + Streamlit + NLP")
