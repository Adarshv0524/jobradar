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

  // ── Filters ─────────────────────────────────────────────────────────────────
  // FIX: Initialize from search payload (were always "all" before, ignoring
  // the remote_preference / experience_level the user already selected)
  let filterRemote    = "all";
  let filterLevel     = "all";
  let sortBy          = "score";
  let searchTerm      = "";

  $: filteredJobs = jobs
    .filter(j => {
      if (filterRemote !== "all" && j.remote_type?.toLowerCase() !== filterRemote) return false;
      if (filterLevel  !== "all" && j.experience_level?.toLowerCase() !== filterLevel)  return false;
      if (searchTerm && !`${j.title} ${j.company} ${j.location ?? ""}`.toLowerCase().includes(searchTerm.toLowerCase())) return false;
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

    // FIX: Auto-initialize filter bar from the search form's selections
    // so results are already pre-filtered when they stream in.
    if (p.remote_preference && p.remote_preference !== "any") {
      filterRemote = p.remote_preference;   // "remote" | "hybrid" | "on-site"
    } else {
      filterRemote = "all";
    }
    if (p.experience_level && p.experience_level !== "") {
      // Map "lead" → "lead" but the filter only has intern/junior/mid/mid-senior/senior
      // so map gracefully
      const lvlMap: Record<string, string> = {
        junior: "junior", intern: "intern", mid: "mid",
        "mid-level": "mid", senior: "senior", lead: "senior",
        staff: "senior", principal: "senior",
      };
      filterLevel = lvlMap[p.experience_level.toLowerCase()] ?? "all";
    } else {
      filterLevel = "all";
    }
    searchTerm = "";

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
            dispatch("trace", trace);   // FIX: was missing the dispatch here
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
          dispatch("siteFound", site);  // FIX: was missing dispatch here too
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
            <span style="color:var(--green);font-size:1.1em;">✓</span>
          {:else if status === "error"}
            <span style="color:var(--red);font-size:1.1em;">✕</span>
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

      <!-- Live traces — real-time agent reasoning inline under status bar -->
      {#if status === "searching" && traces.length > 0}
        <div style="
          margin-top:8px;padding-top:8px;border-top:1px solid var(--border);
          font-family:'JetBrains Mono',monospace;font-size:10px;
          max-height:72px;overflow:hidden;
        ">
          {#each traces.slice(-3) as t}
            <div style="color:var(--text-dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
              {t.icon} {t.message}
            </div>
          {/each}
        </div>
      {/if}
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

      <!-- FIX: Remote filter shows active state when auto-set from search -->
      <select
        bind:value={filterRemote}
        class="bg-surface border border-border rounded px-2 py-1.5 text-xs font-mono
               focus:outline-none {filterRemote !== 'all' ? 'text-amber border-amber/40' : 'text-dim'}"
      >
        <option value="all">All modes</option>
        <option value="remote">Remote</option>
        <option value="hybrid">Hybrid</option>
        <option value="on-site">On-site</option>
      </select>

      <!-- FIX: Level filter shows active state when auto-set from search -->
      <select
        bind:value={filterLevel}
        class="bg-surface border border-border rounded px-2 py-1.5 text-xs font-mono
               focus:outline-none {filterLevel !== 'all' ? 'text-amber border-amber/40' : 'text-dim'}"
      >
        <option value="all">All levels</option>
        <option value="intern">Intern</option>
        <option value="junior">Junior</option>
        <option value="mid">Mid</option>
        <option value="mid-senior">Mid-Senior</option>
        <option value="senior">Senior</option>
      </select>

      <!-- Result count badge -->
      <span style="
        font-size:10px;font-family:'JetBrains Mono',monospace;
        color:var(--text-dim);white-space:nowrap;
      ">
        {filteredJobs.length} / {jobs.length} shown
        {#if filterRemote !== "all" || filterLevel !== "all" || searchTerm}
          — <button
            on:click={() => { filterRemote="all"; filterLevel="all"; searchTerm=""; }}
            style="color:var(--accent);background:none;border:none;cursor:pointer;font-size:10px;font-family:inherit;"
          >clear filters</button>
        {/if}
      </span>

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
          {sessionId}
          on:feedback={(e) => handleFeedback(job.id, e.detail.rating)}
        />
      {/each}
    </div>
  {:else if status === "idle"}
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