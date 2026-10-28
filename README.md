# ProjectScope AI — Universal Project Similarity Finder

> **An intelligent, cross-domain project similarity detection platform that identifies conceptual alignment, shared objectives, and problem statement overlaps—even when implementation frameworks, technologies, titles, and descriptions are entirely distinct.**

---

## 1. Architectural Overview & Concept

Academic institutions, research laboratories, and hackathons encounter thousands of project submissions every year. Traditional duplicate checkers rely on text keyword matching, token n-grams, or shallow TF-IDF metrics. Consequently, they fail in two major ways:
1. **False Negatives:** When two projects share an identical objective but use different technologies (e.g., *"Attendance Management Using Fingerprint Sensors"* vs. *"Smart Attendance Tracking Using Facial Recognition"*), keyword tools fail to see that they solve the exact same problem.
2. **False Positives:** When two projects share a common industry or domain (e.g., *"Online Shopping Website"* vs. *"AI-Based Product Recommendation Engine"*), keyword matchers flag them as duplicates simply because both contain e-commerce terms, despite having completely distinct core goals.

**ProjectScope AI** solves this by decomposing every submission into structured conceptual dimensions and computing a multi-vector weighted semantic representation powered by `sentence-transformers` (`all-MiniLM-L6-v2`) and cosine similarity.

---

## 2. Multi-Signal AI Comparison Engine

### Multi-Vector Weighting Strategy
Rather than computing a single naive cosine similarity over the concatenated text, ProjectScope AI breaks projects down into five distinct orthogonal dimensions. Core objectives and problem statements receive the highest weights, while technology choices receive a low weight:

| Dimension | Default Weight | Description |
| :--- | :---: | :--- |
| **Core Objective** | **35%** | What the project fundamentally seeks to achieve and deliver. |
| **Problem Statement** | **25%** | The exact bottleneck, pain point, or deficiency being addressed. |
| **Functional & Target Users** | **20%** | Intended beneficiaries, user interactions, and expected operational outcomes. |
| **Methodology & Architecture** | **12%** | High-level system design and algorithmic approach. |
| **Technologies & Tech Stack** | **8%** | Specific libraries, languages, and hardware components used. |

$$\text{Overall Similarity} = \frac{\sum (w_i \cdot \text{Sim}_i)}{\sum w_i}$$

*Note: Weights and classification thresholds are completely configurable at runtime by administrators via the Admin Console or `.env` configuration.*

### Suggested Classifications & Thresholds
- **Very High Conceptual Similarity** ($\ge 0.75$ or Objective $\ge 0.78$ & Problem $\ge 0.65$): Projects solve identical problems with near-identical target outcomes.
- **High Conceptual Similarity** ($\ge 0.60$ or Objective $\ge 0.65$): Strong alignment in problem statement and primary deliverable.
- **Partial or Moderate Similarity** ($\ge 0.45$): Shared methodology or overlapping secondary capabilities.
- **Related Domain, Different Objective**: Projects operate in the same domain but solve distinctly different problems (e.g., E-Commerce store vs. Recommender).
- **Low Similarity** ($< 0.35$): Minimal conceptual overlap.
- **Duplicate Alert** ($\ge 0.92$): Flagged for human administrator review.

---

## 3. Project Directory Structure

```text
project/
├── backend/
│   ├── app/
│   │   ├── models/             # SQLAlchemy ORM models (User, Project, Embedding, ComparisonLog, Feedback, AdminAction)
│   │   ├── routers/            # FastAPI route handlers
│   │   │   ├── auth.py         # JWT authentication & user roles
│   │   │   ├── projects.py     # Project listing, detail, submission & duplicate guards
│   │   │   ├── similarity.py   # Multi-vector similarity search & side-by-side compare
│   │   │   ├── admin.py        # Admin review, duplicate merge & weight configuration
│   │   │   ├── stats.py        # System usage analytics and domain distribution
│   │   │   └── discovery.py    # arXiv & GitHub external discovery endpoint
│   │   ├── schemas/            # Pydantic v2 validation models
│   │   ├── services/           # Core AI & business logic
│   │   │   ├── embedding_service.py    # SentenceTransformers ('all-MiniLM-L6-v2') & cosine similarity
│   │   │   ├── nlp_extractor.py        # Text aspect normalization & fallback parsing
│   │   │   ├── similarity_engine.py    # Candidate retrieval, weighted evaluation & ranking
│   │   │   ├── explanation_service.py  # Transparent reasoning generator (similarities & differences)
│   │   │   ├── auth_service.py         # Password hashing & JWT bearer tokens
│   │   │   └── external_discovery.py   # Live arXiv and GitHub API integration with offline fallback
│   │   ├── config.py           # Application settings and environment variables
│   │   ├── database.py         # SQLite / PostgreSQL engine & session setup
│   │   ├── main.py             # FastAPI entrypoint & production static frontend mount
│   │   └── seed_data.py        # 15 foundational multi-domain projects & precomputed vectors
│   ├── tests/
│   │   ├── test_api.py         # Endpoint integration tests (Auth, Projects, Similarity)
│   │   ├── test_similarity.py  # 8 core behavioral unit tests
│   │   └── evaluate_engine.py  # Benchmark evaluation script (Precision@K, MRR, Accuracy)
│   ├── init_db.py              # CLI database migration & seed initialization script
│   ├── requirements.txt        # Backend Python dependencies
│   └── .env.example            # Backend environment template
├── frontend/
│   ├── dist/                   # Production-compiled Vite/React bundle (served by FastAPI)
│   ├── src/
│   │   ├── components/         # Reusable UI components
│   │   │   ├── Navbar.tsx      # Responsive header with theme toggle & user status
│   │   │   ├── Footer.tsx      # Academic platform footer
│   │   │   ├── ProjectCard.tsx # Project card with badges & quick actions
│   │   │   ├── SimilarityGauge.tsx # Circular gauge with color-coded classification
│   │   │   ├── BreakdownChart.tsx  # Horizontal progress bars for the 5 score components
│   │   │   ├── SideBySideCompare.tsx # Field-by-field deep comparison table
│   │   │   ├── WeightConfigModal.tsx # Runtime weight & threshold sliders
│   │   │   └── ProtectedRoute.tsx    # Role-based route guard
│   │   ├── context/
│   │   │   └── AuthContext.tsx # Authentication state & role management
│   │   ├── pages/
│   │   │   ├── HomePage.tsx            # Landing page with DB stats & concept explanation
│   │   │   ├── CheckerPage.tsx         # Interactive similarity checker with demo presets
│   │   │   ├── ExplorerPage.tsx        # Project database browser with filters & pair compare
│   │   │   ├── SubmitPage.tsx          # Project submission with automated duplicate check
│   │   │   ├── AdminPage.tsx           # Administrator review console with Recharts analytics
│   │   │   ├── ExternalDiscoveryPage.tsx # arXiv & GitHub discovery with provenance banners
│   │   │   └── AuthPage.tsx            # Login and registration interface
│   │   ├── services/
│   │   │   └── api.ts          # Axios API client with bearer token interceptor
│   │   ├── types/
│   │   │   └── index.ts        # TypeScript data definitions
│   │   ├── App.tsx             # Route definitions
│   │   ├── index.css           # Tailwind CSS directives
│   │   └── main.tsx            # React DOM root
│   ├── package.json            # Node.js dependencies
│   ├── tailwind.config.js      # Tailwind theme configuration
│   └── vite.config.ts          # Vite bundler configuration
├── projectscope.db             # Local SQLite database pre-seeded with 15 projects
├── .env.example                # Root environment template
└── README.md                   # This documentation
```

---

## 4. Getting Started & Installation

### Prerequisites
- Python 3.9+
- (Optional for frontend development): Node.js 18+ and npm. *Note: The frontend is already pre-compiled into `frontend/dist/` and served directly by FastAPI.*

### Step 1: Install Backend Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 2: Initialize Database and Load Seed Data
The system includes an automated initialization script that creates tables, registers default users, and computes embeddings for 15 multi-domain projects:
```bash
python backend/init_db.py
```

### Step 3: Start the Application
Run the FastAPI application from the project root:
```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Once started, open your web browser to:
- **Full Application UI:** `http://127.0.0.1:8000`
- **Interactive OpenAPI Docs:** `http://127.0.0.1:8000/docs`

---

## 5. Default Demonstration Accounts

The database comes pre-seeded with two ready-to-use user accounts:

| Role | Username | Password | Access Capabilities |
| :--- | :--- | :--- | :--- |
| **Administrator** | `admin` | `AdminPass123!` | Full access: review flagged duplicates, approve/reject submissions, merge records, configure AI weights & thresholds, view system charts. |
| **Student / Researcher** | `student` | `StudentPass123!` | Standard access: submit new projects, query similarity engine, browse indexed records. |

---

## 6. How the 3 Key Scenarios Work

In the **Project Similarity Checker** (`/checker`), click any of the one-click demo presets to observe the engine's behavior:

### Scenario 1: Same Goal, Different Technologies
- **Input:** *"Smart Attendance Tracking Using Facial Recognition"*
- **Matched DB Record:** *"Attendance Management System Using Fingerprint Recognition"*
- **Result:** **High / Very High Conceptual Similarity** (~70-85% goal similarity).
- **Explanation:** The engine recognizes that despite using OpenCV/PyTorch vs. Arduino/Fingerprint sensors, both projects share the core objective of automated classroom attendance tracking and proxy prevention.

### Scenario 2: Cross-Domain Conceptual Overlap
- **Input:** *"AI-Based River Pollution Forecasting Using Sensor Data"* (Domain: IoT / Telemetry)
- **Matched DB Record:** *"Water Quality Prediction Using Machine Learning"* (Domain: Environmental Science)
- **Result:** **High Conceptual Similarity** (~65-75% overall alignment).
- **Explanation:** The engine detects that both systems analyze water quality parameters to anticipate contamination events, recognizing functional synergy across differing engineering domains.

### Scenario 3: Same Domain, Different Objectives
- **Input:** *"Online Shopping Portal with Merchant Checkout"* (Domain: Web & Mobile)
- **Matched DB Record:** *"AI-Based Product Recommendation System for E-Commerce"* (Domain: AI & ML)
- **Result:** **Related domain, different objective** or **Partial similarity** (Goal similarity < 45%).
- **Explanation:** The system does not classify them as duplicates merely because both involve e-commerce; it correctly separates storefront order management from personalized recommendation algorithms.

---

## 7. Running Tests and Benchmark Evaluation

### Automated PyTest Suite (14 Tests)
The test suite validates API endpoints and tests the 8 required AI similarity conditions:
```bash
python -m pytest backend/tests -v
```
All 14 tests pass:
1. `test_different_wording_same_goal`
2. `test_different_implementation_same_goal`
3. `test_similar_domain_different_goals`
4. `test_cross_domain_overlapping_objectives`
5. `test_short_incomplete_description`
6. `test_exact_duplicate_detection`
7. `test_no_meaningful_similarity`
8. `test_empty_database_handling`
9. API login, registration, project list, detail, similarity check, and pair comparison.

### Engine Benchmark Evaluation Script
Measure Precision@1, Precision@3, MRR, and classification accuracy against a benchmark dataset:
```bash
python backend/tests/evaluate_engine.py
```
**Results:**
- **Precision@1 (Top-1 Hit):** 80.0%
- **Precision@3 (Top-3 Hit):** 80.0%
- **Mean Reciprocal Rank (MRR):** 0.800 / 1.000
- **Classification Accuracy:** 60.0% (conservative probabilistic calibration)

---

## 8. Database & Production Scaling (PostgreSQL + pgvector)

The default development setup uses SQLite with high-performance in-memory vector cosine similarity calculation.

For production deployments with >10,000 projects:
1. Provision a PostgreSQL 15+ database and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
2. Update `DATABASE_URL` in your `.env` file:
   ```env
   DATABASE_URL="postgresql://user:password@localhost:5432/projectscope"
   ```
3. Run `python backend/init_db.py` to create the schema and index embeddings using HNSW or IVFFlat for sub-10ms approximate nearest neighbor (ANN) retrieval.

---

## 9. External Discovery Feature

The platform includes an **External Discovery** service (`/discovery`) that interfaces with:
- **arXiv API:** Searches peer-reviewed open-access computer science and engineering preprints.
- **GitHub REST API:** Searches open-source repositories for implementation benchmarks.
- **Graceful Fallback:** If internet access is offline or rate limits apply, the service gracefully switches to curated offline discovery entries, preventing application errors and clearly annotating data provenance.

---

## 10. License

Released under the MIT License. Developed for academic integrity, research coordination, and universal project analysis.
