# 🎯 AI Job Hunter & ATS Screener

> *Disclaimer from the builder:* I’m a mechanical engineer by trade—normally I design hardware, calculate tolerances, and mess with CAD/PLM systems. This is my very first attempt at building and open-sourcing a full Python application. If a few gears squeak or a bolt comes loose, bear with me! Hopefully, this saves you dozens of hours in your job hunt.

---

An automated, cross-platform job hunting utility built with Streamlit, JobSpy, and Google Gemini. 

It sweeps major aggregators (LinkedIn, Indeed, Glassdoor, Google Jobs) and discovers direct corporate ATS listings (Workday, SuccessFactors, Taleo, Greenhouse, Lever). It then pairs the listings with semantic AI gap analysis against your resume—or functions entirely in a lightning-fast "Search-Only" mode without requiring an API key.

> **Roadmap Note:** This current release (`v1.0`) runs using the **Gemini Cloud API** (with dynamic model fetching and automatic retry batching). **Version 2.0** is actively in development and will introduce a 100% offline, local AI workflow via **LM Studio / Apple MLX / Ollama** for zero-cost, private inference on your own hardware.

---

## ⚡ Key Features

- **Dual-Mode Workflow:**
  - **Mode A (Search Only):** No API key or resume required. Aggregates and deduplicates listings into a clean table and downloadable CSV.
  - **Mode B (AI Screener):** Provide your Gemini API key and upload your resume (PDF). The app dynamically pulls your account's available models (e.g., `gemini-3.8-flash`), evaluates candidate fit, scores jobs (0–100), and outputs exact tool and skill gaps.
- **Direct ATS Discovery:** Uses targeted dorking to uncover hidden job postings on corporate portals before aggregators even index them.
- **Spreadsheet Ready:** Exports structured CSV files formatted with active `=HYPERLINK()` formulas ready for Google Sheets or Excel tracking.

---

## 🛠️️ Step-by-Step Installation & Setup

### Prerequisites
- Python 3.10+ installed on your system.
- Git installed.

---

### macOS Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/AI-Job-Scraper.git
   cd AI-Job-Scraper