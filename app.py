import io
import json
import os
import pandas as pd
import streamlit as st
from duckduckgo_search import DDGS
from google import genai
from google.genai import types
from jobspy import scrape_jobs
from pypdf import PdfReader

st.set_page_config(page_title="AI Job Hunter & ATS Screener", layout="wide")

# --- Dynamic Model Fetching ---
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_available_models(token: str) -> list:
    try:
        client = genai.Client(api_key=token)
        models = []
        for m in client.models.list():
            name = m.name.replace("models/", "")
            if "gemini" in name and "vision" not in name:
                models.append(name)
        return sorted(list(set(models)))
    except Exception:
        return []

# --- UI Header ---
st.title("🎯 AI Job Hunter & ATS Screener")
st.caption("Scrape ATS portals and aggregators (<30 days). Run without an API key to just search, or add a key + resume for AI screening.")

# --- Sidebar: Configuration & Model Selection ---
st.sidebar.header("Configuration")
env_api_key = os.getenv("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input(
    "Gemini API Key (Optional)", 
    value=env_api_key, 
    type="password", 
    help="Leave blank to use 'Search Only' mode. Enter key for AI Gap Analysis."
)

available_models = []
selected_model = None

if api_key:
    available_models = fetch_available_models(api_key)
    if available_models:
        default_idx = next((i for i, m in enumerate(available_models) if "3.8-flash" in m), 0)
        selected_model = st.sidebar.selectbox("🤖 Select Gemini Model", options=available_models, index=default_idx)
    else:
        st.sidebar.error("⚠️️ Invalid API Key or no models found for this account.")

# --- Step 1: Resume Upload ---
st.subheader("Step 1: Upload Resume (Optional)")
if not api_key:
    st.info("ℹ️ Enter a valid Gemini API key in the sidebar to unlock AI Resume Analysis.")

uploaded_file = st.file_uploader("Upload your resume (PDF) [Required for AI Analysis]", type=["pdf"])
resume_text = ""
if uploaded_file is not None:
    try:
        pdf_reader = PdfReader(uploaded_file)
        resume_text = "\n".join([page.extract_text() or "" for page in pdf_reader.pages])
        st.success(f"Resume loaded successfully ({len(resume_text)} characters extracted).")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")

# --- Step 2: Target Criteria ---
st.subheader("Step 2: Target Criteria")
col1, col2, col3 = st.columns(3)
with col1:
    industry = st.text_input("Target Industry", placeholder="e.g., Semiconductor Equipment, Aerospace")
with col2:
    role = st.text_input("Target Role", placeholder="e.g., Mechanical Design Engineer")
with col3:
    location = st.text_input("Target Location", placeholder="e.g., Bengaluru, Europe, Remote")

# --- Step 3: Search Functions ---
def search_aggregators(target_role: str, target_location: str, target_industry: str) -> list:
    query = f"{target_role} {target_industry}".strip()
    try:
        # Removed country_dict constraint so it respects the raw location string globally
        jobs_df = scrape_jobs(
            site_name=["linkedin", "indeed", "glassdoor", "google"],
            search_term=query,
            location=target_location,
            results_wanted=15,
            hours_old=720,
            linkedin_fetch_description=True 
        )
        if jobs_df is None or jobs_df.empty:
            return []
        
        jobs_df = jobs_df.astype(object).fillna("Not Specified")
        results = []
        for _, row in jobs_df.iterrows():
            desc = str(row.get("description", ""))
            if desc == "Not Specified" or desc == "nan" or len(desc) < 100:
                continue
                
            results.append({
                "title": str(row.get("title", "Unknown Title")),
                "company": str(row.get("company", "Unknown Company")),
                "location": str(row.get("location", target_location)),
                "job_url": str(row.get("job_url", "")),
                "description": desc[:1500], 
                "date_posted": str(row.get("date_posted", "Within 30 days")),
                "source": "Aggregator"
            })
        return results
    except Exception as err:
        st.warning(f"Aggregator sweep warning: {err}")
        return []

def search_direct_ats(target_role: str, target_location: str, target_industry: str) -> list:
    dork_query = (
        f'(site:myworkdayjobs.com OR site:successfactors.com OR site:taleo.net OR site:greenhouse.io OR site:lever.co) '
        f'"{target_role}" {target_location} {target_industry}'
    ).strip()
    results = []
    try:
        with DDGS() as ddgs:
            raw_hits = ddgs.text(dork_query, max_results=10, safesearch="off")
            for hit in raw_hits:
                desc = hit.get("body", "")
                if len(desc) < 50: 
                    continue
                    
                results.append({
                    "title": hit.get("title", "Direct ATS Role"),
                    "company": "Direct ATS Listing (See URL)",
                    "location": target_location,
                    "job_url": hit.get("href", ""),
                    "description": desc[:1500],
                    "date_posted": "Within 30 days",
                    "source": "Direct ATS"
                })
        return results
    except Exception as err:
        st.warning(f"Direct ATS search warning: {err}")
        return []

import time
import json
from google import genai
from google.genai import types

# --- Step 4: Gemini Evaluation Engine ---
def evaluate_jobs(candidate_resume: str, job_listings: list, api_token: str, target_model: str) -> list:
    client = genai.Client(api_key=api_token)

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "Tier": {"type": "string", "description": "1 OEM, 2 Supplier, or 3 Agency"},
                "Score": {"type": "integer", "description": "Semantic match score (0 to 100)"},
                "Title": {"type": "string"},
                "Job_Link": {"type": "string"},
                "Key_Skills": {"type": "string", "description": "Top 4-6 required technical competencies"},
                "Company": {"type": "string"},
                "Location": {"type": "string"},
                "Posted": {"type": "string"},
                "Date_Status": {"type": "string", "description": "Fresh (<7 days), Recent (7-20 days), or Older (20+ days)"},
                "Notes": {"type": "string", "description": "Concrete gaps: CAD/PLM tool differences, domain mismatch, or YOE delta"}
            },
            "required": ["Tier", "Score", "Title", "Job_Link", "Key_Skills", "Company", "Location", "Posted", "Date_Status", "Notes"]
        }
    }

    all_evaluations = []
    batch_size = 5  # Process 5 jobs at a time to keep payloads light
    
    # Process jobs in chunks
    for i in range(0, len(job_listings), batch_size):
        batch = job_listings[i:i + batch_size]
        
        formatted_jds = ""
        for idx, item in enumerate(batch):
            formatted_jds += (
                f"\n--- JOB {i + idx + 1} ---\n"
                f"Title: {item['title']}\n"
                f"Company: {item['company']}\n"
                f"URL: {item['job_url']}\n"
                f"Description: {item['description'][:1000]}\n" # Aggressively cap description length
            )

        prompt = f"""
        You are an expert technical recruitment screener and career consultant.
        Evaluate the Candidate's Resume against each scraped job posting below.
        Pay strict attention to CAD/PLM tooling equivalence, manufacturing domains, and experience level.
        In 'Notes', call out exact tool differences and domain gaps.
        
        --- CANDIDATE RESUME ---
        {candidate_resume}
        
        --- JOB LISTINGS ---
        {formatted_jds}
        """

        # Automatic Retry Loop (Exponential Backoff)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=target_model,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.1
                    )
                )
                batch_results = json.loads(response.text)
                all_evaluations.extend(batch_results)
                break # Success, break out of retry loop
                
            except Exception as e:
                if "503" in str(e) or "429" in str(e):
                    if attempt < max_retries - 1:
                        sleep_time = 2 ** attempt # Waits 1s, then 2s
                        time.sleep(sleep_time)
                        continue
                # If it's a different error or we ran out of retries, log it and move to next batch
                st.warning(f"Failed to analyze a batch of {len(batch)} jobs after {max_retries} attempts. Error: {e}")
                break 

    return all_evaluations

# --- Step 5: Execution Trigger & Output ---
st.divider()

is_ai_mode = bool(api_key and available_models and resume_text)
btn_text = "🚀 Find & Analyze Jobs (AI Screened)" if is_ai_mode else "🔍 Search & Scrape Jobs Only"
btn_type = "primary" if is_ai_mode else "secondary"

if st.button(btn_text, type=btn_type, use_container_width=True):
    if not role or not location:
        st.error("Please specify both a target role and target location.")
    elif api_key and available_models and not resume_text:
        st.error("API Key detected. Please upload a PDF resume to run the AI gap analysis, or clear the API key field to run in Search-Only mode.")
    else:
        # Variables to hold results outside the status block
        df = pd.DataFrame()
        has_results = False

        with st.status("Executing Search Pipeline...", expanded=True) as status:
            st.write("🔍 Sweeping LinkedIn, Indeed, Glassdoor, and Google Jobs (< 30 days)...")
            agg_results = search_aggregators(role, location, industry)

            st.write("🌐 Sweeping Direct ATS portals (Workday, SuccessFactors, Taleo, Greenhouse)...")
            ats_results = search_direct_ats(role, location, industry)

            combined_jobs = agg_results + ats_results
            st.write(f"📊 Gathered {len(combined_jobs)} candidate listings. Removing duplicates...")

            unique_jobs = []
            seen_urls = set()
            for j in combined_jobs:
                url = j.get("job_url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_jobs.append(j)

            if not unique_jobs:
                status.update(label="No open positions found matching criteria.", state="error")
            else:
                has_results = True
                if is_ai_mode:
                    st.write(f"🧠 Running {selected_model} gap analysis on {len(unique_jobs)} distinct openings...")
                    try:
                        evaluation = evaluate_jobs(resume_text, unique_jobs, api_key, selected_model)
                        status.update(label="AI Analysis complete!", state="complete", expanded=False)
                        df = pd.DataFrame(evaluation)
                    except Exception as ai_err:
                        status.update(label=f"AI Engine Error: {ai_err}", state="error")
                        has_results = False
                else:
                    status.update(label="Scraping complete!", state="complete", expanded=False)
                    df = pd.DataFrame(unique_jobs)

        # OUTPUT RENDERED OUTSIDE THE STATUS BLOCK
        if has_results and not df.empty:
            if is_ai_mode:
                df.sort_values(by="Score", ascending=False, inplace=True)
                df["Role (click to open)"] = df.apply(
                    lambda r: f'=HYPERLINK("{r["Job_Link"]}", "{r["Title"]}")', axis=1
                )
                
                df["Score (0–100)"] = df["Score"]
                df["Key skills needed"] = df["Key_Skills"]
                df["Date status"] = df["Date_Status"]
                df["Notes / gaps"] = df["Notes"]
                df["My status"] = ""

                final_display_cols = [
                    "Tier", "Score (0–100)", "Title", "Job_Link", 
                    "Key skills needed", "Company", "Location", 
                    "Posted", "Date status", "Notes / gaps"
                ]
                
                st.subheader("📋 AI Evaluation Results")
                st.dataframe(
                    df[final_display_cols],
                    column_config={
                        "Job_Link": st.column_config.LinkColumn("Direct Link", display_text="Open Job"),
                        "Score (0–100)": st.column_config.ProgressColumn("Match Score", min_value=0, max_value=100)
                    },
                    use_container_width=True,
                    hide_index=True
                )

                export_cols = [
                    "Tier", "Score (0–100)", "Role (click to open)", 
                    "Key skills needed", "Company", "Location", 
                    "Posted", "Date status", "Notes / gaps", "My status"
                ]
                csv_buffer = io.StringIO()
                df[export_cols].to_csv(csv_buffer, index=False)

                st.download_button(
                    label="📥 Download AI Analysis CSV for Google Sheets",
                    data=csv_buffer.getvalue(),
                    file_name="Job_Search_AI_Results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                df["Role (click to open)"] = df.apply(
                    lambda r: f'=HYPERLINK("{r["job_url"]}", "{r["title"]}")', axis=1
                )
                
                st.subheader("📋 Raw Scraped Jobs")
                st.dataframe(
                    df[["title", "company", "location", "source", "date_posted", "job_url"]],
                    column_config={
                        "job_url": st.column_config.LinkColumn("Direct Link", display_text="Open Job")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                export_cols = ["Role (click to open)", "company", "location", "source", "date_posted"]
                csv_buffer = io.StringIO()
                df[export_cols].to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="📥 Download Scraped Jobs CSV",
                    data=csv_buffer.getvalue(),
                    file_name="Raw_Scraped_Jobs.csv",
                    mime="text/csv",
                    use_container_width=True
                )