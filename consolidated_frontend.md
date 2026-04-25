# Consolidated Code: `frontend`

## ` .env`

```dotenv
# Backend
PORT=8000
HOST=0.0.0.0

# Frontend — points to backend
PUBLIC_API_URL=http://localhost:8000

# Optional: sentence-transformers model cache dir
SENTENCE_TRANSFORMERS_HOME=~/.cache/torch/sentence_transformers
```

## `astro.config.mjs`

```javascript
import { defineConfig } from "astro/config";
import svelte           from "@astrojs/svelte";
import tailwind         from "@astrojs/tailwind";

export default defineConfig({
  integrations: [svelte(), tailwind()],
  server: { port: 4321 },
  vite: {
    define: {
      "import.meta.env.PUBLIC_API_URL":
        JSON.stringify(process.env.PUBLIC_API_URL || "http://localhost:8000"),
    },
  },
});
```

## `package.json`

```json
{
  "name": "jobradar-frontend",
  "type": "module",
  "version": "1.0.0",
  "scripts": {
    "dev":   "astro dev",
    "build": "astro build",
    "preview": "astro preview"
  },
  "dependencies": {
    "astro":           "^4.8.0",
    "@astrojs/svelte": "^5.4.0",
    "@astrojs/tailwind": "^5.1.0",
    "svelte":          "^4.2.17",
    "tailwindcss":     "^3.4.3"
  }
}
```

## `src/components/AppShell.svelte`

```svelte
<script lang="ts">
  import SearchForm   from "./SearchForm.svelte";
  import JobBoard     from "./JobBoard.svelte";
  import SiteSidebar  from "./SiteSidebar.svelte";
  import type { SearchPayload } from "../lib/api";

  let boardRef: JobBoard;
  let sidebarRef: SiteSidebar;
  let pendingPayload: SearchPayload | null = null;
  let isSearching = false;

  function handleSearch(e: CustomEvent<SearchPayload>) {
    pendingPayload = e.detail;
  }

  function handleSearchStart() { isSearching = true; }
  function handleSearchDone()  { isSearching = false; }

  $: if (pendingPayload && boardRef) {
    boardRef.startNewSearch(pendingPayload);
    pendingPayload = null;
  }
</script>

<div style="display:flex;gap:20px;align-items:flex-start;">

  <!-- ── Left Sidebar ─────────────────────────────────────── -->
  <aside style="
    width: 280px;
    flex-shrink: 0;
    position: sticky;
    top: 72px;
    max-height: calc(100vh - 92px);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  ">
    <SiteSidebar bind:this={sidebarRef} {isSearching} />
  </aside>

  <!-- ── Main Content ─────────────────────────────────────── -->
  <main style="flex:1;min-width:0;display:flex;flex-direction:column;gap:16px;">
    <SearchForm on:search={handleSearch} />
    <JobBoard
      bind:this={boardRef}
      on:searchStart={handleSearchStart}
      on:searchDone={handleSearchDone}
      on:planUpdate={(e) => sidebarRef?.handlePlanUpdate(e.detail)}
      on:trace={(e) => sidebarRef?.addTrace(e.detail)}
      on:siteFound={(e) => sidebarRef?.markSiteFound(e.detail)}
    />
  </main>

</div>
```

## `src/components/JobBoard.svelte`

```svelte
<script lang="ts">
  import JobCard from "./JobCard.svelte";
  import { createEventDispatcher } from "svelte";
  import {
    createSSEStream, getSessionJobs, submitFeedback,
    type Job, type SearchPayload,
  } from "../lib/api";
  import { startSearch } from "../lib/api";

  const dispatch = createEventDispatcher();

  export let payload: SearchPayload | null = null;

  let sessionId    = "";
  let jobs: Job[]  = [];
  let status: "idle" | "searching" | "done" | "error" = "idle";
  let statusMsg    = "";
  let jobsFound    = 0;
  let pagesCrawled = 0;
  let tokensTotal   = 0;
  let highQuality  = 0;
  let sitesCount   = 0;
  let traces: { icon: string; message: string; ts: number }[] = [];
  let sitesDiscovered: Record<string, number> = {};
  let es: EventSource | null = null;

  // Filters
  let filterRemote    = "all";
  let filterLevel     = "all";
  let sortBy          = "score";
  let searchTerm      = "";

  $: filteredJobs = jobs
    .filter(j => {
      if (filterRemote !== "all" && j.remote_type?.toLowerCase() !== filterRemote) return false;
      if (filterLevel  !== "all" && j.experience_level?.toLowerCase() !== filterLevel)  return false;
      if (searchTerm && !`${j.title} ${j.company}`.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      return true;
    })
    .sort((a, b) => {
      if (sortBy === "score")   return b.score - a.score;
      if (sortBy === "recent") {
        const da = a.posted_date || "", db = b.posted_date || "";
        return db.localeCompare(da);
      }
      if (sortBy === "company") return (a.company || "").localeCompare(b.company || "");
      return 0;
    });

  export async function startNewSearch(p: SearchPayload) {
    // Cancel previous stream
    if (es) { es.close(); es = null; }

    jobs       = [];
    status     = "searching";
    dispatch("searchStart");
    statusMsg  = "Starting search…";
    jobsFound  = 0;
    pagesCrawled = 0;
    tokensTotal  = 0;
    highQuality = 0;
    sitesCount  = 0;
    traces      = [];
    sitesDiscovered = {};

    try {
      const res = await startSearch(p);
      sessionId = res.session_id;

      es = createSSEStream(
        sessionId,
        (job) => {
          jobs = [...jobs, job];
          jobsFound = jobs.length;
        },
        (msg) => { statusMsg = msg; },
        (stats) => {
          status     = "done";
          dispatch("searchDone");
          statusMsg  = `Found ${stats.jobs_found} jobs across ${stats.pages_crawled} pages`;
          jobsFound  = stats.jobs_found   ?? jobsFound;
          pagesCrawled = stats.pages_crawled ?? pagesCrawled;
        },
        (err) => {
          status    = "error";
          statusMsg = err;
        },
        (trace) => {
          if (trace.data) {
            traces = [...traces, { icon: trace.icon || "·", message: trace.message || "", ts: Date.now() }];
            traces = traces.slice(-20);
          }
        },
        (stats) => {
          jobsFound    = stats.jobs_found ?? jobsFound;
          pagesCrawled = stats.pages_crawled ?? pagesCrawled;
          tokensTotal  = stats.tokens_total ?? tokensTotal;
          highQuality = stats.high_quality ?? highQuality;
          sitesCount  = stats.sites_count ?? sitesCount;
        },
        (site) => {
          if (site.domain && site.jobs_count) {
            sitesDiscovered[site.domain] = (sitesDiscovered[site.domain] || 0) + site.jobs_count;
            sitesDiscovered = sitesDiscovered;
          }
        },
      );
    } catch (err: any) {
      status    = "error";
      statusMsg = err.message;
    }
  }

  async function handleFeedback(jobId: string, rating: 1 | -1) {
    if (!sessionId) return;
    try {
      await submitFeedback(jobId, sessionId, rating);
    } catch { /* silent */ }
  }

  // Watch payload prop for external triggers
  $: if (payload) startNewSearch(payload);
</script>

<div class="space-y-4">

  <!-- ── Status bar ─────────────────────────────────── -->
  {#if status !== "idle"}
    <div class="bg-card border border-border rounded-lg px-4 py-3">
      <div class="flex items-center justify-between flex-wrap gap-2">
        <div class="flex items-center gap-3">
          {#if status === "searching"}
            <span class="pulse-dot"></span>
          {:else if status === "done"}
            <span class="text-emerald text-lg">✓</span>
          {:else if status === "error"}
            <span class="text-rose text-lg">✕</span>
          {/if}
          <span class="text-sm font-500 text-text">{statusMsg}</span>
        </div>
        <div class="flex items-center gap-4">
          {#each [["Jobs", jobsFound], ["Pages", pagesCrawled], ["Sites", sitesCount], ["★ Quality", highQuality]] as [label, val]}
            <div class="text-center">
              <div class="text-base font-700 text-text font-mono">{val}</div>
              <div class="text-[10px] text-muted">{label}</div>
            </div>
          {/each}
        </div>
      </div>
    </div>
  {/if}

  <!-- Filter + sort bar -->
  {#if jobs.length > 0}
    <div class="flex flex-wrap gap-2 items-center pb-2 border-b border-border">
      <input
        bind:value={searchTerm}
        placeholder="Filter results…"
        class="bg-surface border border-border rounded px-3 py-1.5 text-xs font-mono
               text-text placeholder:text-muted focus:outline-none focus:border-amber/40 w-44"
      />

      <select
        bind:value={filterRemote}
        class="bg-surface border border-border rounded px-2 py-1.5 text-xs font-mono text-dim
               focus:outline-none"
      >
        <option value="all">All modes</option>
        <option value="remote">Remote</option>
        <option value="hybrid">Hybrid</option>
        <option value="on-site">On-site</option>
      </select>

      <select
        bind:value={filterLevel}
        class="bg-surface border border-border rounded px-2 py-1.5 text-xs font-mono text-dim
               focus:outline-none"
      >
        <option value="all">All levels</option>
        <option value="intern">Intern</option>
        <option value="junior">Junior</option>
        <option value="mid">Mid</option>
        <option value="mid-senior">Mid-Senior</option>
        <option value="senior">Senior</option>
      </select>

      <div class="ml-auto flex gap-1">
        {#each [["score","Relevance"],["recent","Recent"],["company","Company"]] as [val, label]}
          <button
            on:click={() => sortBy = val}
            class="text-xs font-mono px-2.5 py-1 rounded border transition-colors {
              sortBy === val
                ? 'border-amber/50 text-amber bg-amber/10'
                : 'border-border text-muted hover:border-dim'
            }"
          >
            {label}
          </button>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Job list -->
  {#if filteredJobs.length > 0}
    <div class="space-y-3">
      {#each filteredJobs as job, i (job.id)}
        <JobCard
          {job}
          index={i}
          on:feedback={(e) => handleFeedback(job.id, e.detail.rating)}
        />
      {/each}
    </div>
  {:else if status === "idle"}
    <!-- Empty state -->
    <div class="text-center py-20 space-y-3">
      <div class="text-4xl">⌖</div>
      <p class="font-display text-2xl font-700 text-dim tracking-wide">START SEARCHING</p>
      <p class="font-mono text-xs text-muted max-w-sm mx-auto">
        Enter a query above. JobRadar will search across job boards, ATS pages,
        and company career sites — ranking by fit, not just keywords.
      </p>
    </div>
  {:else if status === "searching" && jobs.length === 0}
    <div class="space-y-2 pt-4">
      {#each Array(4) as _, i}
        <div
          class="bg-card border border-border rounded-lg p-4 animate-pulse"
          style="animation-delay: {i * 150}ms"
        >
          <div class="flex gap-3">
            <div class="flex-1 space-y-2">
              <div class="h-3 bg-border rounded w-24"></div>
              <div class="h-5 bg-border rounded w-64"></div>
              <div class="h-3 bg-border rounded w-48"></div>
              <div class="flex gap-1 mt-2">
                {#each Array(4) as _}
                  <div class="h-4 bg-border rounded w-16"></div>
                {/each}
              </div>
            </div>
            <div class="flex flex-col gap-2">
              <div class="h-8 w-12 bg-border rounded"></div>
              <div class="h-7 w-20 bg-border rounded"></div>
            </div>
          </div>
        </div>
      {/each}
    </div>
  {:else if status === "done" && filteredJobs.length === 0}
    <div class="text-center py-16">
      <p class="font-display text-xl text-dim">No results match current filters</p>
      <button
        on:click={() => { filterRemote="all"; filterLevel="all"; searchTerm=""; }}
        class="mt-2 font-mono text-xs text-amber hover:underline"
      >
        Clear filters
      </button>
    </div>
  {/if}

</div>
```

## `src/components/JobCard.svelte`

```svelte
<script lang="ts">
  import { submitFeedback, type Job } from "../lib/api";

  export let job:       Job;
  export let index:     number = 0;
  export let sessionId: string = "";

  let expanded  = false;
  let myRating: 1 | -1 | 0 = 0;

  async function rate(r: 1 | -1) {
    if (!sessionId || !job.id) return;
    myRating = myRating === r ? 0 : r;
    if (myRating !== 0) await submitFeedback(job.id, sessionId, r);
  }

  $: scorePercent = Math.round((job.score ?? 0) * 100);
  $: scoreColor =
    scorePercent >= 70 ? "var(--green)" :
    scorePercent >= 45 ? "var(--accent)" :
                         "var(--text-lo)";

  $: remoteLabel = {
    remote:   "Remote",
    hybrid:   "Hybrid",
    "on-site":"On-site",
  }[job.remote_type?.toLowerCase() ?? ""] ?? job.remote_type ?? "";

  $: remoteBadgeClass = {
    remote:   "badge-green",
    hybrid:   "badge-blue",
    "on-site": "badge-neutral",
  }[job.remote_type?.toLowerCase() ?? ""] ?? "badge-neutral";

  function fmt(d: string): string {
    if (!d) return "";
    try { return new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric" }); }
    catch { return d; }
  }

  $: sourceLabel =
    job.source_type === "jsonld"       ? "structured" :
    job.source_type === "detail_page"  ? "detail page" :
    job.source_type?.startsWith("api") ? "API" : "scraped";
</script>

<article
  class="card card-hover fade-in"
  style="
    padding: 16px;
    position: relative;
    animation-delay: {Math.min(index * 40, 400)}ms;
  "
>
  <!-- Score bar (top edge) -->
  <div style="
    position:absolute;top:0;left:0;right:0;height:2px;
    border-radius:8px 8px 0 0;overflow:hidden;
  ">
    <div style="
      height:100%;
      width:{scorePercent}%;
      background:{scoreColor};
      border-radius:inherit;
      transition:width 0.7s;
    "/>
  </div>

  <!-- ── Main row ─────────────────────────────────── -->
  <div style="display:flex;gap:14px;align-items:flex-start;">

    <!-- Left: info -->
    <div style="flex:1;min-width:0;">

      <!-- Meta line -->
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:6px;">
        <span class="mono" style="font-size:10px;color:var(--text-dim);">#{String(index+1).padStart(2,"0")}</span>
        <span class="badge {remoteBadgeClass}">{remoteLabel}</span>
        {#if job.posted_date}
          <span class="mono" style="font-size:10px;color:var(--text-dim);">{fmt(job.posted_date)}</span>
        {/if}
        <span class="mono" style="font-size:10px;color:var(--text-dim);margin-left:auto;">{sourceLabel}</span>
      </div>

      <!-- Title -->
      <h3 style="
        font-size:15px;font-weight:600;color:var(--text-hi);
        margin:0 0 4px;line-height:1.3;
        overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
      ">{job.title || "Untitled Role"}</h3>

      <!-- Company / location / salary -->
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:10px;">
        <span style="font-size:13px;font-weight:500;color:var(--text);">{job.company || "Unknown"}</span>
        {#if job.location}
          <span style="color:var(--text-dim);">·</span>
          <span style="font-size:12px;color:var(--text-lo);">{job.location}</span>
        {/if}
        {#if job.salary}
          <span style="color:var(--text-dim);">·</span>
          <span class="badge badge-accent">{job.salary}</span>
        {/if}
      </div>

      <!-- Skills -->
      {#if job.skills?.length}
        <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">
          {#each job.skills.slice(0,8) as skill}
            <span class="skill-tag">{skill}</span>
          {/each}
          {#if job.skills.length > 8}
            <span class="badge badge-neutral">+{job.skills.length - 8}</span>
          {/if}
        </div>
      {/if}

      <!-- Match reason -->
      {#if job.match_reasons?.length && !expanded}
        <p style="font-size:11px;color:var(--green);font-family:'JetBrains Mono',monospace;">
          ✓ {job.match_reasons[0]}
        </p>
      {/if}
    </div>

    <!-- Right: score + actions -->
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;flex-shrink:0;">

      <!-- Score pill -->
      <div class="mono" style="
        font-size:13px;font-weight:600;
        padding:4px 10px;border-radius:5px;
        border:1px solid {scoreColor}40;
        background:{scoreColor}12;
        color:{scoreColor};
      ">{scorePercent}%</div>

      <!-- Feedback buttons -->
      <div style="display:flex;gap:4px;">
        <button
          on:click={() => rate(1)}
          style="
            width:28px;height:28px;border-radius:5px;cursor:pointer;
            border:1px solid {myRating===1 ? 'var(--green)' : 'var(--border)'};
            background:{myRating===1 ? 'var(--green-lo)' : 'transparent'};
            color:{myRating===1 ? 'var(--green)' : 'var(--text-dim)'};
            display:flex;align-items:center;justify-content:center;
            font-size:14px;transition:all 0.15s;
          "
          title="Good match"
        >↑</button>
        <button
          on:click={() => rate(-1)}
          style="
            width:28px;height:28px;border-radius:5px;cursor:pointer;
            border:1px solid {myRating===-1 ? 'var(--red)' : 'var(--border)'};
            background:{myRating===-1 ? 'var(--red-lo)' : 'transparent'};
            color:{myRating===-1 ? 'var(--red)' : 'var(--text-dim)'};
            display:flex;align-items:center;justify-content:center;
            font-size:14px;transition:all 0.15s;
          "
          title="Not relevant"
        >↓</button>
      </div>

      <!-- Apply -->
      <a
        href={job.apply_url} target="_blank" rel="noopener noreferrer"
        style="
          font-size:11px;font-weight:600;font-family:'JetBrains Mono',monospace;
          padding:5px 12px;border-radius:5px;
          border:1px solid var(--accent);
          background:var(--accent-lo);
          color:var(--accent);
          text-decoration:none;
          transition:all 0.15s;letter-spacing:0.05em;
        "
        on:mouseover={e => e.currentTarget.style.background='var(--accent-hi)'}
        on:mouseout={e => e.currentTarget.style.background='var(--accent-lo)'}
      >APPLY →</a>
    </div>
  </div>

  <!-- Expand toggle -->
  <button
    on:click={() => expanded = !expanded}
    style="
      margin-top:8px;font-size:11px;font-family:'JetBrains Mono',monospace;
      color:var(--text-dim);background:none;border:none;cursor:pointer;
      padding:0;transition:color 0.15s;
    "
    on:mouseover={e => e.currentTarget.style.color='var(--text-lo)'}
    on:mouseout={e => e.currentTarget.style.color='var(--text-dim)'}
  >{expanded ? "▲ hide details" : "▼ show details"}</button>

  <!-- Expanded panel -->
  {#if expanded}
    <div class="fade-in" style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border);">

      <!-- Score breakdown -->
      {#if job.score_breakdown && Object.keys(job.score_breakdown).length}
        <p style="font-size:10px;font-weight:600;color:var(--text-dim);letter-spacing:0.07em;margin-bottom:8px;">SIGNAL BREAKDOWN</p>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:12px;">
          {#each Object.entries(job.score_breakdown) as [key, val]}
            {@const n = Number(val)}
            <div class="surface" style="padding:8px;text-align:center;">
              <p style="font-size:10px;color:var(--text-dim);letter-spacing:0.06em;text-transform:uppercase;margin:0 0 2px;">{key}</p>
              <p class="mono" style="font-size:14px;font-weight:600;margin:0;color:{n>0.5?'var(--green)':n>0.25?'var(--accent)':'var(--text-dim)'}">
                {Math.round(n*100)}%
              </p>
            </div>
          {/each}
        </div>
      {/if}

      <!-- Match reasons -->
      {#if job.match_reasons?.length}
        <p style="font-size:10px;font-weight:600;color:var(--text-dim);letter-spacing:0.07em;margin-bottom:6px;">WHY IT MATCHED</p>
        <div style="margin-bottom:12px;">
          {#each job.match_reasons as r}
            <p style="font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--green);margin:2px 0;">✓ {r}</p>
          {/each}
        </div>
      {/if}

      <!-- Description -->
      {#if job.description}
        <p style="font-size:10px;font-weight:600;color:var(--text-dim);letter-spacing:0.07em;margin-bottom:6px;">DESCRIPTION</p>
        <p style="font-size:12px;color:var(--text-lo);line-height:1.6;margin:0 0 12px;">
          {job.description.slice(0,700)}{job.description.length>700?"…":""}
        </p>
      {/if}

      <div style="display:flex;gap:12px;font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--text-dim);">
        <span>src: {job.source_domain}</span>
        {#if job.experience_level}
          <span>· {job.experience_level}</span>
        {/if}
      </div>
    </div>
  {/if}
</article>
```

## `src/components/SearchForm.svelte`

```svelte
<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { uploadResume, type SearchPayload } from "../lib/api";

  const dispatch = createEventDispatcher<{ search: SearchPayload }>();

  let query            = "";
  let role             = "";
  let location         = "";
  let remotePreference = "any";
  let experienceLevel  = "";
  let minSalary        = "";
  let skills           = "";
  let summary          = "";
  let negatives: string[] = [];
  let uploading        = false;
  let uploadMsg        = "";
  let showAdvanced     = false;

  const NEGATIVE_OPTIONS = [
    { key: "no_agency",      label: "No agencies" },
    { key: "no_crypto",      label: "No crypto/web3" },
    { key: "no_sales",       label: "No sales roles" },
    { key: "no_unpaid",      label: "No unpaid work" },
    { key: "no_relocation",  label: "No relocation req." },
  ];

  function toggleNeg(key: string) {
    negatives = negatives.includes(key)
      ? negatives.filter(k => k !== key)
      : [...negatives, key];
  }

  async function handleResumeUpload(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    uploading = true; uploadMsg = "";
    try {
      const r = await uploadResume(file);
      skills  = r.skills.join(", ");
      summary = r.summary;
      if (r.experience_level) experienceLevel = r.experience_level;
      uploadMsg = `✓ ${r.skills.length} skills extracted`;
    } catch {
      uploadMsg = "⚠ Upload failed";
    } finally { uploading = false; }
  }

  function handleSubmit() {
    if (!query.trim()) return;
    dispatch("search", {
      query,
      role:               role || query,
      location,
      remote_preference:  remotePreference,
      experience_level:   experienceLevel,
      min_salary:         minSalary ? parseInt(minSalary) * 1000 : 0,
      negatives,
      skills:             skills.split(",").map(s => s.trim()).filter(Boolean),
      summary,
      experience_summary: "",
      role_target:        role,
    });
  }
</script>

<div class="card" style="padding:20px;">

  <!-- Search bar row -->
  <div style="display:flex;gap:8px;margin-bottom:16px;">
    <input
      class="input"
      bind:value={query}
      placeholder="e.g. Senior Python Engineer, Data Scientist, DevOps Lead…"
      on:keydown={e => e.key === "Enter" && handleSubmit()}
      style="flex:1;"
    />
    <button class="btn btn-primary" on:click={handleSubmit} disabled={!query.trim()}>
      Search
    </button>
  </div>

  <!-- Quick filters row -->
  <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">

    <select class="input" bind:value={remotePreference} style="width:auto;padding-right:28px;">
      <option value="any">Any work type</option>
      <option value="remote">Remote only</option>
      <option value="hybrid">Hybrid</option>
      <option value="on-site">On-site</option>
    </select>

    <select class="input" bind:value={experienceLevel} style="width:auto;padding-right:28px;">
      <option value="">Any level</option>
      <option value="junior">Junior</option>
      <option value="mid">Mid-level</option>
      <option value="senior">Senior</option>
      <option value="lead">Lead / Staff</option>
    </select>

    <input
      class="input"
      bind:value={location}
      placeholder="Location (optional)"
      style="width:160px;"
    />

    <button
      class="btn btn-ghost"
      on:click={() => showAdvanced = !showAdvanced}
      style="font-size:12px;"
    >
      {showAdvanced ? "▲" : "▼"} Advanced
    </button>
  </div>

  <!-- Advanced panel -->
  {#if showAdvanced}
    <div class="fade-in" style="margin-top:16px;padding-top:16px;border-top:1px solid var(--border);">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">

        <div>
          <label class="label">Min Salary (K USD)</label>
          <input class="input" type="number" bind:value={minSalary} placeholder="e.g. 80" />
        </div>

        <div>
          <label class="label">Skills (comma-separated)</label>
          <input class="input" bind:value={skills} placeholder="python, react, aws…" />
        </div>
      </div>

      <div style="margin-bottom:12px;">
        <label class="label">Profile Summary (improves ranking)</label>
        <textarea
          class="input" bind:value={summary} rows="2"
          placeholder="Brief description of your background and goals…"
          style="resize:vertical;"
        ></textarea>
      </div>

      <!-- Resume upload -->
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <label class="btn btn-ghost" style="cursor:pointer;font-size:12px;">
          {uploading ? "Uploading…" : "↑ Upload Resume (PDF / TXT)"}
          <input
            type="file" accept=".pdf,.txt" style="display:none;"
            on:change={handleResumeUpload} disabled={uploading}
          />
        </label>
        {#if uploadMsg}
          <span style="font-size:12px;color:var(--green);font-family:'JetBrains Mono',monospace;">
            {uploadMsg}
          </span>
        {/if}
      </div>

      <!-- Negatives -->
      <div>
        <label class="label">Exclude</label>
        <div style="display:flex;flex-wrap:wrap;gap:6px;">
          {#each NEGATIVE_OPTIONS as opt}
            <button
              on:click={() => toggleNeg(opt.key)}
              style="
                padding:4px 10px;border-radius:5px;font-size:11px;font-weight:500;
                cursor:pointer;transition:all 0.15s;border:1px solid;
                background:{negatives.includes(opt.key) ? 'var(--red-lo)' : 'transparent'};
                border-color:{negatives.includes(opt.key) ? 'rgba(248,113,113,0.3)' : 'var(--border)'};
                color:{negatives.includes(opt.key) ? 'var(--red)' : 'var(--text-lo)'};
              "
            >{opt.label}</button>
          {/each}
        </div>
      </div>
    </div>
  {/if}
</div>
```

## `src/components/SiteSidebar.svelte`

```svelte
<script lang="ts">
  export let isSearching = false;

  type SiteStatus = "pending" | "running" | "done" | "failed" | "skipped";
  type SitePlan = {
    site_key: string;
    site_label: string;
    status: SiteStatus;
    jobs_found: number;
    pages_done: number;
    pages_total: number;
  };

  type TraceEntry = {
    level: string;
    icon: string;
    message: string;
    ts: number;
  };

  let sites: Map<string, SitePlan>  = new Map();
  let traces: TraceEntry[]          = [];
  let activeTab: "sites" | "reasoning" = "sites";
  let totalJobs = 0;
  let totalSites = 0;

  export function handlePlanUpdate(data: SitePlan) {
    sites = new Map(sites.set(data.site_key, data));
    if (data.status === "done" && data.jobs_found > 0) {
      totalJobs += data.jobs_found;
    }
    totalSites = [...sites.values()].filter(s => s.status === "done" || s.status === "running").length;
  }

  export function addTrace(trace: TraceEntry) {
    traces = [trace, ...traces].slice(0, 60);
  }

  export function markSiteFound(data: { domain: string; jobs_count: number }) {
    // Legacy support for "site" events
  }

  $: runningCount = [...sites.values()].filter(s => s.status === "running").length;
  $: doneCount    = [...sites.values()].filter(s => s.status === "done").length;
  $: failedCount  = [...sites.values()].filter(s => s.status === "failed").length;

  function statusIcon(s: SiteStatus): string {
    return { pending: "○", running: "◌", done: "●", failed: "✕", skipped: "–" }[s] ?? "○";
  }
  function statusColor(s: SiteStatus): string {
    return {
      pending: "var(--text-dim)",
      running: "var(--accent)",
      done:    "var(--green)",
      failed:  "var(--red)",
      skipped: "var(--text-dim)",
    }[s] ?? "var(--text-dim)";
  }

  function traceColor(level: string): string {
    return {
      plan: "var(--accent)", search: "var(--blue)", fetch: "var(--text-lo)",
      extract: "var(--green)", api: "var(--blue)", llm: "#a78bfa",
      eval: "var(--accent)", warn: "var(--red)", crawl: "var(--text-lo)",
      done: "var(--green)",
    }[level] ?? "var(--text-lo)";
  }
</script>

<!-- ── Header ─────────────────────────────────────── -->
<div class="card" style="padding:14px;">
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
    <span style="font-size:12px;font-weight:600;color:var(--text-hi);letter-spacing:0.05em;">CRAWL STATUS</span>
    {#if isSearching}
      <div style="display:flex;align-items:center;gap:5px;">
        <div class="pulse-dot"></div>
        <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--green);">LIVE</span>
      </div>
    {/if}
  </div>

  <!-- Stats row -->
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:14px;">
    {#each [
      {label:"Done",   val: doneCount,   color: "var(--green)"},
      {label:"Active", val: runningCount, color: "var(--accent)"},
      {label:"Failed", val: failedCount,  color: "var(--red)"},
    ] as stat}
      <div style="
        background:var(--surface);border:1px solid var(--border);
        border-radius:6px;padding:8px;text-align:center;
      ">
        <div style="font-size:18px;font-weight:700;color:{stat.color};font-family:'JetBrains Mono',monospace;">{stat.val}</div>
        <div style="font-size:10px;color:var(--text-dim);margin-top:2px;">{stat.label}</div>
      </div>
    {/each}
  </div>

  <!-- Tabs -->
  <div style="display:flex;gap:4px;margin-bottom:12px;">
    {#each [["sites","Sites"],["reasoning","Reasoning"]] as [tab, label]}
      <button
        on:click={() => activeTab = tab}
        style="
          flex:1; padding:5px 8px; border-radius:5px; font-size:11px; font-weight:500;
          cursor:pointer; transition:all 0.15s; border:1px solid;
          background:{activeTab===tab ? 'var(--elevated)' : 'transparent'};
          border-color:{activeTab===tab ? 'var(--border-hi)' : 'transparent'};
          color:{activeTab===tab ? 'var(--text)' : 'var(--text-dim)'};
        "
      >{label}</button>
    {/each}
  </div>

  <!-- ── Sites tab ──────────────────────────────────── -->
  {#if activeTab === "sites"}
    <div style="display:flex;flex-direction:column;gap:3px;max-height:420px;overflow-y:auto;">
      {#if sites.size === 0}
        <p style="font-size:12px;color:var(--text-dim);text-align:center;padding:20px 0;">
          {isSearching ? "Building crawl plan…" : "Start a search to see sites"}
        </p>
      {:else}
        {#each [...sites.values()].sort((a,b) => {
          const order = {running:0,done:1,pending:2,failed:3,skipped:4};
          return (order[a.status]??5) - (order[b.status]??5);
        }) as site (site.site_key)}
          <div class="slide-in" style="
            display:flex;align-items:center;gap:8px;
            padding:5px 6px; border-radius:5px;
            background:{site.status==='running' ? 'rgba(245,158,11,0.06)' : 'transparent'};
            border:1px solid {site.status==='running' ? 'rgba(245,158,11,0.15)' : 'transparent'};
            transition:all 0.2s;
          ">
            <span style="
              font-size:10px; width:10px; flex-shrink:0;
              color:{statusColor(site.status)};
              font-family:'JetBrains Mono',monospace;
            ">{statusIcon(site.status)}</span>
            <span style="flex:1;font-size:11px;color:{site.status==='pending' ? 'var(--text-dim)' : 'var(--text)' };min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">
              {site.site_label}
            </span>
            {#if site.status === "done" && site.jobs_found > 0}
              <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--green);flex-shrink:0;">
                +{site.jobs_found}
              </span>
            {:else if site.status === "running"}
              <span style="font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--accent);flex-shrink:0;">
                p.{site.pages_done}
              </span>
            {/if}
          </div>
        {/each}
      {/if}
    </div>

  <!-- ── Reasoning tab ──────────────────────────────── -->
  {:else}
    <div style="display:flex;flex-direction:column;gap:4px;max-height:420px;overflow-y:auto;">
      {#if traces.length === 0}
        <p style="font-size:12px;color:var(--text-dim);text-align:center;padding:20px 0;">
          {isSearching ? "Agent is thinking…" : "Agent reasoning will appear here"}
        </p>
      {:else}
        {#each traces as trace (trace.ts)}
          <div class="slide-in" style="padding:4px 6px;border-radius:4px;">
            <div style="display:flex;gap:6px;align-items:flex-start;">
              <span style="font-size:12px;flex-shrink:0;margin-top:1px;">{trace.icon}</span>
              <p style="font-size:11px;color:var(--text-lo);line-height:1.5;margin:0;word-break:break-word;">
                <span style="color:{traceColor(trace.level)};font-weight:500;">[{trace.level}]</span>
                {" "}{trace.message}
              </p>
            </div>
          </div>
        {/each}
      {/if}
    </div>
  {/if}
</div>
```

## `src/env.d.ts`

```typescript
/// <reference path="../.astro/types.d.ts" />
```

## `src/lib/api.ts`

```typescript
const API_URL = import.meta.env.PUBLIC_API_URL || "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface Job {
  id:               string;
  session_id:       string;
  title:            string;
  company:          string;
  location:         string;
  remote_type:      string;
  experience_level: string;
  salary:           string;
  skills:           string[];
  description:      string;
  apply_url:        string;
  source_domain:    string;
  source_type:      string;
  posted_date:      string;
  score:            number;
  score_breakdown:  Record<string, number>;
  match_reasons:    string[];
  reject_reasons:   string[];
  avg_feedback:     number;
}

export interface Session {
  id:            string;
  query:         string;
  status:        string;
  jobs_found:    number;
  pages_crawled: number;
  created_at:    string;
}

export interface SearchPayload {
  query:             string;
  role?:             string;
  location?:         string;
  remote_preference?: string;
  experience_level?: string;
  min_salary?:       number;
  negatives?:        string[];
  skills?:           string[];
  summary?:          string;
  experience_summary?: string;
  role_target?:      string;
}

export interface SSEEvent {
  type:          "job" | "progress" | "done" | "error" | "trace" | "stats" | "site" | "plan_update";
  job?:          Job;
  message?:      string;
  query_index?:  number;
  jobs_found?:   number;
  pages_crawled?: number;
  level?:        string;
  icon?:         string;
  data?:         Record<string, unknown>;
  tokens_total?: number;
  tokens_prompt?: number;
  tokens_comp?:  number;
  high_quality?: number;
  sites_count?:  number;
  llm_enabled?:  boolean;
  llm_guided?:   boolean;
  model?:        string;
  queries_used?: number;
  domain?:       string;
  jobs_count?:   number;
}

// ── API calls ─────────────────────────────────────────────────────────────────

export async function startSearch(payload: SearchPayload): Promise<{ session_id: string }> {
  const res = await fetch(`${API_URL}/api/search`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Search failed: ${res.statusText}`);
  return res.json();
}

export function createSSEStream(
  sessionId: string,
  onJob:      (job: Job)       => void,
  onProgress: (msg: string)    => void,
  onDone:     (stats: SSEEvent) => void,
  onError:    (err: string)    => void,
  onTrace?:      (trace: SSEEvent) => void,
  onStats?:      (stats: SSEEvent) => void,
  onSite?:       (site: SSEEvent) => void,
  onPlanUpdate?: (plan: SSEEvent) => void,
): EventSource {
  const es = new EventSource(`${API_URL}/api/search/${sessionId}/stream`);

  es.onmessage = (e: MessageEvent) => {
    try {
      const event: SSEEvent = JSON.parse(e.data);
      if (event.type === "job"         && event.job)     onJob(event.job);
      if (event.type === "progress"    && event.message) onProgress(event.message);
      if (event.type === "trace"       && onTrace)       onTrace(event);
      if (event.type === "stats"       && onStats)       onStats(event);
      if (event.type === "site"        && onSite)        onSite(event);
      if (event.type === "plan_update" && onPlanUpdate)  onPlanUpdate(event);
      if (event.type === "done")  { onDone(event);                  es.close(); }
      if (event.type === "error") { onError(event.message || "Unknown error"); es.close(); }
    } catch { /* malformed event */ }
  };

  es.onerror = () => { onError("Connection lost"); es.close(); };
  return es;
}

export async function getSessionJobs(
  sessionId: string
): Promise<{ session: Session; jobs: Job[]; count: number }> {
  const res = await fetch(`${API_URL}/api/jobs/${sessionId}`);
  if (!res.ok) throw new Error("Failed to fetch jobs");
  return res.json();
}

export async function submitFeedback(
  jobId: string, sessionId: string, rating: 1 | -1
): Promise<void> {
  await fetch(`${API_URL}/api/feedback`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ job_id: jobId, session_id: sessionId, rating }),
  });
}

export async function getSessions(): Promise<{ sessions: Session[] }> {
  const res = await fetch(`${API_URL}/api/sessions`);
  if (!res.ok) throw new Error("Failed to fetch sessions");
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${API_URL}/api/sessions/${sessionId}`, { method: "DELETE" });
}

export async function uploadResume(file: File): Promise<{
  skills: string[];
  experience_level: string;
  summary: string;
}> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_URL}/api/upload-resume`, { method: "POST", body: form });
  if (!res.ok) throw new Error("Upload failed");
  return res.json();
}
```

## `src/pages/index.astro`

```astro
---
import AppShell from "../components/AppShell.svelte";
---
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>JobRadar — Deep Job Discovery</title>
  <meta name="description" content="AI-powered job discovery. Search 60+ sites, 100 pages deep." />
  <link rel="icon" type="image/svg+xml"
    href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><text y='26' font-size='26'>◎</text></svg>" />
</head>
<body>
  <!-- Top nav bar -->
  <header style="
    position: sticky; top: 0; z-index: 50;
    background: rgba(13,14,16,0.85);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #2a2d36;
    padding: 0 24px;
    height: 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  ">
    <div style="display:flex;align-items:center;gap:10px;">
      <span style="font-size:18px;color:#f59e0b;">◎</span>
      <span style="font-size:15px;font-weight:700;color:#f1f2f4;letter-spacing:0.02em;">JobRadar</span>
      <span style="font-size:11px;color:#555b6e;font-family:'JetBrains Mono',monospace;margin-left:4px;">v2.0</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      {["60+ Sites","Deep Crawl","AI Ranked"].map(l => (
        <span style="
          font-family:'JetBrains Mono',monospace;
          font-size:10px;
          color:#8891a4;
          border:1px solid #2a2d36;
          border-radius:4px;
          padding:2px 8px;
          letter-spacing:0.05em;
        ">{l}</span>
      ))}
    </div>
  </header>

  <main style="max-width:1400px;margin:0 auto;padding:24px 20px;">
    <AppShell client:load />
  </main>
</body>
</html>
```

## `src/styles/global.css`

```css
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap");

@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --bg:        #0d0e10;
  --surface:   #13151a;
  --elevated:  #1a1d24;
  --card:      #1e2028;
  --border:    #2a2d36;
  --border-hi: #3a3d48;
  --accent:    #f59e0b;
  --accent-lo: rgba(245,158,11,0.12);
  --accent-hi: rgba(245,158,11,0.25);
  --green:     #22c55e;
  --green-lo:  rgba(34,197,94,0.12);
  --blue:      #60a5fa;
  --blue-lo:   rgba(96,165,250,0.12);
  --red:       #f87171;
  --red-lo:    rgba(248,113,113,0.12);
  --text-hi:   #f1f2f4;
  --text:      #c9ccd4;
  --text-lo:   #8891a4;
  --text-dim:  #555b6e;
}

html { scroll-behavior: smooth; }

body {
  background-color: var(--bg);
  color:            var(--text);
  font-family:      "Inter", system-ui, -apple-system, sans-serif;
  font-size:        14px;
  line-height:      1.6;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Scrollbar */
::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-dim); }

/* ── Utility classes ── */

.card {
  background:   var(--card);
  border:       1px solid var(--border);
  border-radius: 8px;
}

.card-hover {
  transition: border-color 0.15s, background 0.15s, box-shadow 0.15s;
}
.card-hover:hover {
  border-color: var(--border-hi);
  background:   var(--elevated);
  box-shadow:   0 4px 24px rgba(0,0,0,0.3);
}

.surface {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
}

.badge {
  display:      inline-flex;
  align-items:  center;
  gap:          4px;
  padding:      2px 8px;
  border-radius: 4px;
  font-size:    11px;
  font-weight:  500;
  font-family:  "JetBrains Mono", monospace;
  letter-spacing: 0.02em;
  border:       1px solid;
}

.badge-accent   { color: var(--accent); border-color: var(--accent-hi); background: var(--accent-lo); }
.badge-green    { color: var(--green);  border-color: rgba(34,197,94,0.3); background: var(--green-lo); }
.badge-blue     { color: var(--blue);   border-color: rgba(96,165,250,0.3); background: var(--blue-lo); }
.badge-neutral  { color: var(--text-lo); border-color: var(--border); background: var(--surface); }
.badge-red      { color: var(--red);    border-color: rgba(248,113,113,0.3); background: var(--red-lo); }

.skill-tag {
  display:      inline-block;
  padding:      2px 7px;
  border-radius: 3px;
  font-size:    11px;
  font-family:  "JetBrains Mono", monospace;
  background:   rgba(96,165,250,0.08);
  border:       1px solid rgba(96,165,250,0.2);
  color:        var(--blue);
}

.score-bar-track {
  height:       2px;
  background:   var(--border);
  border-radius: 2px;
  overflow:     hidden;
}
.score-bar-fill {
  height:       100%;
  border-radius: 2px;
  transition:   width 0.7s cubic-bezier(0.4,0,0.2,1);
}

.btn {
  display:      inline-flex;
  align-items:  center;
  gap:          6px;
  padding:      7px 14px;
  border-radius: 6px;
  font-size:    13px;
  font-weight:  500;
  cursor:       pointer;
  transition:   all 0.15s;
  border:       1px solid;
  font-family:  inherit;
}

.btn-primary {
  background:   var(--accent);
  border-color: var(--accent);
  color:        #0d0e10;
  font-weight:  600;
}
.btn-primary:hover {
  background:   #fbbf24;
  border-color: #fbbf24;
}
.btn-primary:disabled {
  opacity: 0.5;
  cursor:  not-allowed;
}

.btn-ghost {
  background:   transparent;
  border-color: var(--border);
  color:        var(--text-lo);
}
.btn-ghost:hover {
  background:   var(--elevated);
  border-color: var(--border-hi);
  color:        var(--text);
}

.input {
  background:   var(--surface);
  border:       1px solid var(--border);
  border-radius: 6px;
  color:        var(--text-hi);
  font-family:  inherit;
  font-size:    14px;
  padding:      8px 12px;
  width:        100%;
  transition:   border-color 0.15s;
  outline:      none;
}
.input::placeholder { color: var(--text-dim); }
.input:focus {
  border-color: var(--accent-hi);
  box-shadow:   0 0 0 3px var(--accent-lo);
}
.input:hover { border-color: var(--border-hi); }

.label {
  font-size:    11px;
  font-weight:  600;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color:        var(--text-lo);
  display:      block;
  margin-bottom: 5px;
}

/* Pulse indicator */
.pulse-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  animation: pulse-anim 2s ease-in-out infinite;
}
@keyframes pulse-anim {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.4; transform: scale(1.4); }
}

/* Fade-in animation */
.fade-in {
  animation: fadeIn 0.3s ease-out both;
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* Slide-in for sidebar items */
.slide-in {
  animation: slideIn 0.25s ease-out both;
}
@keyframes slideIn {
  from { opacity: 0; transform: translateX(-6px); }
  to   { opacity: 1; transform: translateX(0); }
}

/* Status dots */
.dot-pending  { color: var(--text-dim); }
.dot-running  { color: var(--accent); }
.dot-done     { color: var(--green); }
.dot-failed   { color: var(--red); }

/* Divider */
.divider {
  height: 1px;
  background: var(--border);
  margin: 16px 0;
}

/* Monospace text */
.mono { font-family: "JetBrains Mono", monospace; }

/* Truncate */
.truncate-2 {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
```

## `tailwind.config.mjs`

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,ts,tsx,svelte}"],
  theme: {
    extend: {
      fontFamily: {
        sans:  ['"Inter"', "system-ui", "sans-serif"],
        mono:  ['"JetBrains Mono"', "monospace"],
      },
      colors: {
        bg:       "#0d0e10",
        surface:  "#13151a",
        elevated: "#1a1d24",
        card:     "#1e2028",
        border:   "#2a2d36",
        "border-hi": "#3a3d48",
        accent:   "#f59e0b",
        green:    "#22c55e",
        blue:     "#60a5fa",
        red:      "#f87171",
        "text-hi": "#f1f2f4",
        text:     "#c9ccd4",
        "text-lo": "#8891a4",
        "text-dim": "#555b6e",
      },
      spacing: { "18": "4.5rem", "88": "22rem" },
      borderRadius: { DEFAULT: "6px" },
    },
  },
  plugins: [],
};
```
