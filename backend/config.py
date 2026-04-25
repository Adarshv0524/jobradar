import os
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

if load_dotenv:
    load_dotenv()

BASE_DIR   = Path(__file__).resolve().parent
ROOT_DIR   = BASE_DIR.parent
DATA_DIR   = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESUME_DIR = DATA_DIR / "resumes"
RESUME_DIR.mkdir(exist_ok=True)

DB_PATH = os.environ.get("JOBRADAR_DB_PATH", str(ROOT_DIR / "jobsearch.db"))

# ── Search limits ─────────────────────────────────────────────────────────────
MAX_QUERIES_PER_SESSION  = int(os.environ.get("MAX_QUERIES",   "200"))
MAX_URLS_PER_QUERY       = int(os.environ.get("MAX_URLS",      "50"))
MAX_PAGES_TOTAL          = int(os.environ.get("MAX_PAGES",     "50000"))
MAX_JOBS_PER_SESSION     = int(os.environ.get("MAX_JOBS",      "5000"))
MIN_QUALITY_JOBS         = int(os.environ.get("MIN_QUALITY",   "50"))
MAX_CONCURRENT_FETCHES   = int(os.environ.get("MAX_CONCURRENT","80"))
MAX_WATCH_CYCLES         = int(os.environ.get("MAX_WATCH_CYCLES", "48"))
WATCH_HEARTBEAT_SEC      = int(os.environ.get("WATCH_HEARTBEAT_SEC", "5"))
SEARCH_EMPTY_WAVE_LIMIT  = int(os.environ.get("SEARCH_EMPTY_WAVE_LIMIT", "6"))

# ── Meta-agent ───────────────────────────────────────────────────────────────
META_AGENT              = os.environ.get("META_AGENT", "true").lower() == "true"
META_AGENT_MAX_STEPS    = int(os.environ.get("META_AGENT_MAX_STEPS", "80"))

# Crawl-first mode: disable free APIs by default
USE_FREE_APIS            = os.environ.get("USE_FREE_APIS", "false").lower() == "true"

# ── HTTP ──────────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT  = 18
MAX_RETRIES      = 3
CRAWL_DELAY      = 0.6   # seconds between requests to same domain

# ── Scoring ───────────────────────────────────────────────────────────────────
# FIX: Was 0.10 — far too low. 10-20% irrelevant jobs were flooding results.
# 0.80 means only strong matches are surfaced (user expectation: 80%+ relevance)
MIN_SCORE_THRESHOLD    = float(os.environ.get("MIN_SCORE_THRESHOLD", "0.80"))
HIGH_QUALITY_THRESHOLD = float(os.environ.get("HIGH_QUALITY_THRESHOLD", "0.90"))

# ── Playwright ────────────────────────────────────────────────────────────────
USE_PLAYWRIGHT         = os.environ.get("USE_PLAYWRIGHT", "true").lower() == "true"
PLAYWRIGHT_POOL_SIZE   = int(os.environ.get("PLAYWRIGHT_POOL", "3"))
PLAYWRIGHT_TIMEOUT     = 25_000   # ms

# Browser-first crawling (selenium-style): attempt rendered fetch before plain HTTP.
BROWSER_FIRST          = os.environ.get("BROWSER_FIRST", "true").lower() == "true"

# ── Selenium (fallback) ──────────────────────────────────────────────────────
USE_SELENIUM           = os.environ.get("USE_SELENIUM", "true").lower() == "true"
SELENIUM_POOL_SIZE     = int(os.environ.get("SELENIUM_POOL", "1"))
SELENIUM_TIMEOUT_MS    = int(os.environ.get("SELENIUM_TIMEOUT_MS", "25000"))
SELENIUM_CHROME_BINARY = os.environ.get("SELENIUM_CHROME_BINARY", "")
SELENIUM_DRIVER_PATH   = os.environ.get("SELENIUM_DRIVER_PATH", "")

# ── ATS platform host fragments ───────────────────────────────────────────────
ATS_DOMAINS = {
    "greenhouse.io", "lever.co", "ashbyhq.com", "workday.com",
    "bamboohr.com", "jobvite.com", "icims.com", "smartrecruiters.com",
    "taleo.net", "successfactors.com", "recruitee.com", "personio.de",
    "breezy.hr", "pinpoint.com", "rippling.com", "dover.com",
    "workable.com", "applytojob.com", "careers.google.com",
    "myworkdayjobs.com", "hire.com", "freshteam.com",
    "jobs.lever.co", "boards.greenhouse.io", "apply.workable.com",
    "jobs.jobvite.com", "careers.icims.com",
}

# ── 60+ Job sites to crawl (key → metadata) ──────────────────────────────────
# Each entry: label, url_template ({query} is replaced), max_pages, needs_js
JOB_SITES: dict[str, dict] = {
    # ── Free JSON APIs ────────────────────────────────────────────────────────
    "remotive":      {"label": "Remotive",       "type": "api", "max_pages": 1},
    "arbeitnow":     {"label": "Arbeitnow",       "type": "api", "max_pages": 5},
    "jobicy":        {"label": "Jobicy",          "type": "api", "max_pages": 1},
    "remoteok":      {"label": "RemoteOK",        "type": "api", "max_pages": 1},
    "himalayas":     {"label": "Himalayas",       "type": "api", "max_pages": 1},
    "themuse":       {"label": "The Muse",        "type": "api", "max_pages": 5},
    "adzuna":        {"label": "Adzuna",          "type": "api", "max_pages": 10},
    "devitjobs":     {"label": "DevITjobs",       "type": "api", "max_pages": 3},

    # ── ATS Aggregators (structured, highly reliable) ─────────────────────────
    "greenhouse":    {
        "label": "Greenhouse ATS",
        "url": "https://boards.greenhouse.io/embed/job_board?for={company}",
        "search": "https://www.google.com/search?q=site:boards.greenhouse.io+{query}",
        "type": "ats", "max_pages": 30,
    },
    "lever":         {
        "label": "Lever ATS",
        "search": "https://www.google.com/search?q=site:jobs.lever.co+{query}",
        "type": "ats", "max_pages": 30,
    },
    "ashby":         {
        "label": "Ashby ATS",
        "search": "https://www.google.com/search?q=site:ashbyhq.com+{query}",
        "type": "ats", "max_pages": 20,
    },
    "workday":       {
        "label": "Workday ATS",
        "search": "https://www.google.com/search?q=site:myworkdayjobs.com+{query}",
        "type": "ats", "max_pages": 20,
    },
    "bamboohr":      {
        "label": "BambooHR ATS",
        "search": "https://www.google.com/search?q=site:bamboohr.com/jobs+{query}",
        "type": "ats", "max_pages": 15,
    },
    "icims":         {
        "label": "iCIMS ATS",
        "search": "https://www.google.com/search?q=site:careers.icims.com+{query}",
        "type": "ats", "max_pages": 15,
    },
    "smartrecruiters": {
        "label": "SmartRecruiters",
        "search": "https://www.google.com/search?q=site:careers.smartrecruiters.com+{query}",
        "type": "ats", "max_pages": 15,
    },
    "workable":      {
        "label": "Workable",
        "url": "https://apply.workable.com/api/v3/jobs?query={query}&limit=100",
        "type": "ats_api", "max_pages": 10,
    },
    "recruitee":     {
        "label": "Recruitee",
        "search": "https://www.google.com/search?q=site:recruitee.com+{query}",
        "type": "ats", "max_pages": 10,
    },
    "breezyhr":      {
        "label": "Breezy HR",
        "search": "https://www.google.com/search?q=site:breezy.hr+{query}",
        "type": "ats", "max_pages": 10,
    },
    "jobvite":       {
        "label": "Jobvite",
        "search": "https://www.google.com/search?q=site:jobs.jobvite.com+{query}",
        "type": "ats", "max_pages": 10,
    },

    # ── Open Job Boards (HTML) ────────────────────────────────────────────────
    "weworkremotely": {
        "label": "We Work Remotely",
        "url": "https://weworkremotely.com/remote-jobs/search?term={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "startup_jobs":  {
        "label": "Startup Jobs",
        "url": "https://startup.jobs/?q={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "ycombinator":   {
        "label": "Y Combinator",
        "url": "https://www.ycombinator.com/jobs?q={query}",
        "type": "html_js", "max_pages": 5,
    },
    "wellfound":     {
        "label": "Wellfound (AngelList)",
        "url": "https://wellfound.com/jobs?q={query}&page={page}",
        "type": "html_js", "max_pages": 20,
    },
    "otta":          {
        "label": "Otta",
        "url": "https://app.otta.com/jobs/search?query={query}",
        "type": "html_js", "max_pages": 10,
    },
    "simplify":      {
        "label": "Simplify Jobs",
        "url": "https://simplify.jobs/jobs?search={query}&page={page}",
        "type": "html_js", "max_pages": 15,
    },
    "builtin":       {
        "label": "Built In",
        "url": "https://builtin.com/jobs/remote?search={query}&page={page}",
        "type": "html_js", "max_pages": 20,
    },
    "remote_co":     {
        "label": "Remote.co",
        "url": "https://remote.co/remote-jobs/search/?search_keywords={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "hn_whoishiring": {
        "label": "HN: Who's Hiring",
        "url": "https://hn.algolia.com/api/v1/search?query={query}+hiring&tags=comment",
        "type": "hn_api", "max_pages": 3,
    },
    "instahyre":     {
        "label": "Instahyre (India)",
        "url": "https://www.instahyre.com/search-jobs/?q={query}&page={page}",
        "type": "html_js", "max_pages": 10,
    },
    "iimjobs":       {
        "label": "IIMJobs (India)",
        "url": "https://www.iimjobs.com/j/{query}.html?page={page}",
        "type": "html", "max_pages": 10,
    },
    "freshersworld": {
        "label": "FreshersWorld (India)",
        "url": "https://www.freshersworld.com/jobs/jobsearch/{query}?page={page}",
        "type": "html", "max_pages": 5,
    },
    "naukri_it": {
        "label": "Naukri IT (India – direct)",
        "url": "https://www.naukri.com/{query}-jobs?jobAge=1",
        "type": "html_js", "max_pages": 5,
    },
    "hirist":        {
        "label": "Hirist (India Tech)",
        "url": "https://www.hirist.tech/search?q={query}&page={page}",
        "type": "html", "max_pages": 5,
    },
    "internshala":   {
        "label": "Internshala (India)",
        "url": "https://internshala.com/jobs/{query}/?page={page}",
        "type": "html_js", "max_pages": 5,
    },
    "cutshort":      {
        "label": "Cutshort (India)",
        "url": "https://cutshort.io/jobs/{query}?page={page}",
        "type": "html_js", "max_pages": 5,
    },
    "crunchboard":   {
        "label": "CrunchBoard",
        "url": "https://www.crunchboard.com/jobs?q={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "eurotechjobs":  {
        "label": "EuroTech Jobs",
        "url": "https://www.eurotechjobs.com/search/?q={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
    "nodesk":        {
        "label": "NoDesk",
        "url": "https://nodesk.co/remote-jobs/{query}/?page={page}",
        "type": "html", "max_pages": 5,
    },
    "jobgether":     {
        "label": "Jobgether",
        "url": "https://jobgether.com/offer?search={query}&page={page}",
        "type": "html", "max_pages": 10,
    },
}

# ── Sites needing Playwright (JS-heavy) ───────────────────────────────────────
JS_HEAVY_SITES = {k for k, v in JOB_SITES.items() if v.get("type") == "html_js"}

# Domains for JS-heavy sites (used by generic pipeline heuristics)
JS_HEAVY_DOMAINS = {
    urlparse(v.get("url", "")).netloc
    for k, v in JOB_SITES.items()
    if k in JS_HEAVY_SITES and v.get("url")
}

# ── Free job API endpoints (no key required) ──────────────────────────────────
FREE_JOB_APIS = {
    "remotive":  "https://remotive.com/api/remote-jobs?search={query}&limit=100",
    "arbeitnow": "https://arbeitnow.com/api/job-board-api?page={page}",
    "jobicy":    "https://jobicy.com/api/v2/remote-jobs?count=100&geo=worldwide&tag={query}",
    "remoteok":  "https://remoteok.com/api?tags={query}",
    "himalayas": "https://himalayas.app/jobs/api?q={query}&limit=100&offset={offset}",
    "themuse":   "https://www.themuse.com/api/public/jobs?page={page}&descending=true",
    "adzuna":    "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}?app_id={app_id}&app_key={app_key}&results_per_page=50&what={query}",
    "devitjobs": "https://devitjobs.us/api/jobsLight",
}

# ── Known aggregator domains to skip ─────────────────────────────────────────
AGGREGATOR_DOMAINS = {
    "indeed.com", "linkedin.com", "glassdoor.com", "naukri.com",
    "timesjobs.com", "monster.com", "ziprecruiter.com", "shine.com",
    "simplyhired.com", "careerbuilder.com", "dice.com",
    "reed.co.uk", "totaljobs.com", "jobsite.co.uk",
}

# ── Spam signals ─────────────────────────────────────────────────────────────
FAKE_SIGNALS = [
    "earn from home", "work from home earn", "be your own boss",
    "multi-level", "multilevel", "mlm", "commission only",
    "unlimited earning potential", "pyramid", "get rich",
    "passive income", "no experience needed but", "make money online",
]

# ── Canonical skill list (EXPANDED with big-data, India-relevant tools) ───────
TECH_SKILLS = [
    # Languages
    "python", "javascript", "typescript", "java", "go", "golang", "rust",
    "c++", "c#", "ruby", "scala", "kotlin", "swift", "php",
    # Frontend
    "react", "vue", "angular", "svelte", "nextjs", "nuxt",
    # Backend frameworks
    "fastapi", "django", "flask", "spring", "rails", "laravel",
    "nodejs", "express", "graphql", "rest", "grpc",
    # DevOps / infra
    "docker", "kubernetes", "k8s", "terraform", "ansible",
    "aws", "gcp", "azure", "cloud", "lambda", "s3",
    # Databases
    "postgresql", "mysql", "sqlite", "mongodb", "redis",
    "elasticsearch", "cassandra", "dynamodb", "hbase",
    # ── Big Data & Data Engineering (FIXED: was missing many) ────────────────
    "kafka", "spark", "apache spark", "pyspark",          # Streaming / compute
    "airflow", "prefect", "dagster", "luigi",              # Orchestration
    "dbt", "dbt core",                                     # Transformation
    "databricks", "delta lake", "iceberg", "hudi",         # Lakehouse
    "snowflake", "bigquery", "redshift", "synapse",        # Cloud DW
    "hadoop", "hive", "presto", "trino", "impala",         # Hadoop ecosystem
    "flink", "storm",                                      # Stream processing
    "aws glue", "azure data factory", "gcp dataflow",      # ETL services
    "nifi", "talend", "informatica",                       # ETL tools
    "great expectations", "deequ",                         # Data quality
    "data catalog", "apache atlas", "collibra",            # Governance
    # ML / AI
    "pytorch", "tensorflow", "sklearn", "pandas", "numpy", "polars",
    "llm", "machine learning", "deep learning", "nlp", "computer vision",
    "mlflow", "mlops", "feature store", "kubeflow",
    # BI / Analytics
    "power bi", "tableau", "looker", "metabase", "superset",
    "excel", "google sheets",
    # General tech
    "sql", "nosql", "microservices", "devops", "sre", "ci/cd", "git",
    "linux", "bash", "data engineering", "data science",
    "celery", "rabbitmq", "protobuf", "openapi",
    "react native", "flutter", "ios", "android",
]

# ── US-only location signals (used in ranker to penalise for India searches) ──
US_ONLY_LOCATION_SIGNALS = [
    "united states", "usa only", "u.s. only", "us only",
    "must be located in the us", "must reside in the us",
    "remote us", "remote - us", "remote (us", "remote us only",
    "north america only", "americas only", "us citizens only",
    # US states / cities that make it unambiguously US
    "new york", "san francisco", "los angeles", "seattle", "boston",
    "chicago", "austin", "atlanta", "denver", "portland", "miami",
    ", ca", ", ny", ", wa", ", tx", ", ma",   # state abbreviations in location
]

# ── India location signals (to boost for India searches) ─────────────────────
INDIA_LOCATION_SIGNALS = [
    "india", "bangalore", "bengaluru", "hyderabad", "pune", "mumbai",
    "chennai", "delhi", "ncr", "noida", "gurgaon", "gurugram",
    "kolkata", "ahmedabad", "jaipur", "kochi", "remote india",
]

# ── CORS ──────────────────────────────────────────────────────────────────────
CORS_ORIGINS = [
    "http://localhost:4321",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:4321",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

# ── User agents ───────────────────────────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
]

# ── Adzuna (optional — get free key at developer.adzuna.com) ──────────────────
ADZUNA_APP_ID  = os.environ.get("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.environ.get("ADZUNA_APP_KEY", "")

# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4.1")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
# For OpenAI-compatible gateways (e.g. Azure OpenAI /openai/v1) that expect `api-key`.
OPENAI_API_KEY_HEADER = os.environ.get("OPENAI_API_KEY_HEADER", "")
