# Sunwa-AI (सुनवाई-AI) — Public Grievance Intelligence & Prioritization Platform

> **Pronunciation:** *Sunwaai* (/sʊn.vɑː.iː/)  
> **Mission:** Multilingual civic complaint intelligence, semantic incident clustering, deterministic priority scoring, emergency hazard detection, and observational intervention tracking.

---

## 🏛️ System Overview

**Sunwa-AI** is a robust civic-technology grievance intelligence platform designed for municipal administration (such as Navi Mumbai Municipal Corporation - NMMC) and civic authorities. It ingests public infrastructure grievances across multiple local languages (English, Hindi, Marathi, and Romanized/code-mixed scripts), deterministically resolves physical landmarks against a curated municipal catalogue, clusters related complaints into spatio-temporal incidents using multilingual semantic embeddings, computes deterministic priority scores with safety escalation overrides, and tracks observational intervention impact over time.

---

## 🌟 Key Features

### 1. Dual Portal Architecture
- **🏛️ Administrative Intelligence Dashboard**:
  - **Executive Overview**: High-level KPIs, priority distribution, category breakdowns, and regional grievance density.
  - **Interactive Incident Map & Hotmap**: Geographic Folium visualization with complaint-density HeatMap layers, priority filtering, and separate unlocated grievance panels.
  - **Semantic Cluster Explorer**: Deep inspection of semantic groupings, cluster cohesion, language compositions, coordinates, and preserved outlier reports.
  - **Priority Queue**: Deterministic priority ranking with transparent formula breakdowns and optional Groq contextual interpretations.
  - **Emergency Hazard Panel**: Dedicated urgent safety detection for life-safety risks (open electrical wires, cave-ins, structural hazards, water contamination) with deterministic overrides.
  - **Grounded Evidence Dossier**: Audit trail citing verbatim citizen text, landmark match metadata, uncertainty analysis, and scoring factors.
  - **Recurrence & Pattern Analysis**: Distinguishes localized incidents, short-term complaint surges, chronic recurring decay, and distributed systemic patterns.
  - **Intervention Tracking**: Logs municipal maintenance actions and evaluates before/after complaint trends across custom time windows with observational disclaimers.
- **📋 Citizen Grievance Portal**: Independent submission interface allowing citizens to log complaints in English, हिंदी, or मराठी with instant reference ID generation and zero personal tracking.

### 2. Full Application Localization (English, Hindi, Marathi)
- Seamlessly switch between **English**, **हिंदी (Hindi)**, and **मराठी (Marathi)** via the high-contrast sidebar selector.
- Complete localization across all 8 dashboard panels and the citizen submission portal.
- Preserves verbatim citizen text in original language and script.

### 3. Multilingual NLP & Deterministic Geocoding
- Classifies grievances across **English, Hindi, Marathi, Romanized Hindi, Romanized Marathi**, and mixed-language text.
- Resolves landmarks using an explicit municipal catalogue (`data/landmarks.csv`) via rapid fuzzy matching, aliases, and substring normalization.
- **Groq LLM Fallback**: If deterministic matching is ambiguous or unlocated, Groq safely selects from valid candidate landmarks.
- **Strict Preservative Policy**: Unlocated reports are preserved at the region level without fabricating GPS coordinates.

### 4. Multilingual Semantic Clustering
- Groups complaints describing the same incident using multilingual sentence embeddings, geographic distance thresholds, and temporal windows.
- Prevents similarity-chain merging across distinct physical landmarks.
- Preserves singletons and outliers so no citizen grievance is obscured.

### 5. Deterministic Priority Engine & Safety Overrides
- **Transparent Mathematical Formula**:
  $$\text{Priority Score } (P) = 0.25S + 0.20A + 0.15D + 0.25R + 0.15T$$
  - **S**: Severity (0–100)
  - **A**: Affected Population (0–100)
  - **D**: Duration (0–100)
  - **R**: Safety Risk (0–100)
  - **T**: Recurrence (0–100)
- **Life Safety Overrides**: Critical hazards trigger mandatory overrides ($P \ge 95.0$, Tier = Critical), preventing ordinary averages from hiding safety risks.
- **Strict LLM Separation**: Groq LLM never calculates, alters, or overrides numerical priority scores. LLM contextual analysis is clearly labeled and cites only valid complaint IDs.

### 6. Observational Intervention Tracking
- Record administrative interventions against specific incident clusters or regional patterns.
- Evaluates complaint volume before and after the intervention date across custom day windows.
- Safely handles zero baselines and missing dates.
- Explicit observational integrity disclaimer (correlation $\neq$ causation).
- Persists records across application restarts to `data/interventions.json`.

---

## 🚀 How to Run the Application

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)

### Installation
1. Navigate to the project directory:
   ```bash
   cd CivicPulse-AI
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Configure Groq API Key:
   Ensure `GROQ_API` (or `GROQ_API_KEY`) is present in your environment or `.env`:
   ```bash
   GROQ_API=your_groq_api_key_here
   ```
   *(Note: The entire application, including emergency hazard detection, deterministic scoring, clustering, and map visualization, is 100% operational without an API key).*

### Running the App
Run the Streamlit application:
```bash
streamlit run app.py
```
- Open `http://localhost:8501` in your browser.
- Toggle between **🏛️ Admin Portal** and **📋 Citizen Portal** from the header dropdown.
- Switch interface languages from the sidebar.

---

## 🧪 Running Automated Tests

Run the full pytest suite (61 unit & integration tests):
```bash
pytest -v
```

All 61 tests cover data loading, schema integrity, fuzzy location resolution, multilingual clustering, priority scoring, safety overrides, intervention persistence, and graceful LLM degradation.

---

## 📂 Project Architecture

```
CivicPulse-AI/
├── app.py                         # Streamlit application entry point (Dual portal, high-contrast sidebar)
├── engine/                        # Core algorithmic & analytical pipeline
│   ├── pipeline.py                # Pipeline orchestrator & submission handler
│   ├── location_resolver.py       # Deterministic landmark geocoder & catalogue
│   ├── complaint_normalizer.py    # Text cleaning, language detection & safety keyword extraction
│   ├── semantic_engine.py         # Multilingual embeddings (MiniLM / TF-IDF fallback)
│   ├── clustering_engine.py       # Spatio-temporal constrained incident clustering
│   ├── priority_engine.py         # Deterministic priority scoring formula & safety overrides
│   ├── pattern_analyzer.py        # Surges, recurring decay, and systemic pattern classification
│   ├── intervention_manager.py    # Before/after observational impact evaluation
│   ├── recommendation_engine.py   # Grounded municipal department routing
│   ├── data_loader.py             # CSV/JSON persistence & schema loaders
│   └── llm_client.py              # Groq API client with safe fallback
├── components/                    # UI dashboard & portal components
│   ├── overview.py                # Executive KPI analytics dashboard
│   ├── incident_map.py            # Geographic Folium map with HeatMap density layer
│   ├── cluster_explorer.py        # Semantic cluster inspection & outlier preservation
│   ├── priority_queue.py          # Deterministic priority queue & Groq contextual briefs
│   ├── emergency_panel.py         # Urgent safety hazard detection & override inspection
│   ├── evidence_panel.py          # Grounded evidence dossier & audit trail
│   ├── pattern_analysis.py        # Recurrence & systemic pattern analyzer
│   ├── intervention_tracking.py   # Intervention creation form & before/after trend charts
│   └── citizen_portal.py          # Multilingual citizen grievance intake form
├── locales/                       # Application internationalization (i18n)
│   ├── i18n.py                    # Localization controller
│   ├── en.json                    # English translations
│   ├── hi.json                    # Hindi translations (हिंदी)
│   └── mr.json                    # Marathi translations (मराठी)
├── models/                        # Dataclass schemas
│   └── schemas.py
├── data/                          # Data persistence
│   ├── complaints.csv             # Citizen complaint records
│   ├── landmarks.csv              # Curated municipal landmark catalogue
│   └── interventions.json         # Administrative intervention records
└── tests/                         # Pytest automated test suites (61 tests)
```
