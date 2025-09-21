# Resume Relevance Checker\n\n## Quick start\n\n```bash\npython -m venv venv\nsource venv/bin/activate\npip install -r backend/requirements.txt\npython -m spacy download en_core_web_sm\n```\n\n### Run backend\n```bash\nuvicorn backend.app.main:app --reload\n```\n\n### Run frontend\n```bash\nstreamlit run streamlit_app.py\n```\n
# example
# cd C:\Users\RVS10\OneDrive\Desktop\resume-relevance-checker
# if you don't have venv, create and activate:
# python -m venv venv
.\venv\Scripts\activate

pip install streamlit PyMuPDF python-docx docx2txt spacy sentence-transformers fuzzywuzzy python-Levenshtein pandas matplotlib
python -m spacy download en_core_web_sm

# then run:
python -m streamlit run app.py