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
  let sortBy          = "live";
  let searchTerm      = "";

  $: filteredJobs = jobs
    .filter(j => {
      if (filterRemote !== "all" && j.remote_type?.toLowerCase() !== filterRemote) return false;
      if (filterLevel  !== "all" && j.experience_level?.toLowerCase() !== filterLevel)  return false;
      if (searchTerm && !`${j.title} ${j.company} ${j.location ?? ""}`.toLowerCase().includes(searchTerm.toLowerCase())) return false;
      return true;
    })
    .sort((a, b) => {
      if (sortBy === "score") return b.score - a.score;
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
      dispatch("session", { sessionId });

      es = createSSEStream(
        sessionId,
        (job) => {
          jobs = [job, ...jobs.filter(existing => existing.id !== job.id)];
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
          traces = [...traces, { icon: trace.icon || "·", message: trace.message || "", ts: Date.now() }];
          traces = traces.slice(-20);
          dispatch("trace", trace);
        },
        (stats) => {
          jobsFound    = stats.jobs_found ?? jobsFound;
          pagesCrawled = stats.pages_crawled ?? pagesCrawled;
          tokensTotal  = stats.tokens_total ?? tokensTotal;
          highQuality = stats.high_quality ?? highQuality;
          sitesCount  = stats.sites_count ?? sitesCount;
          dispatch("stats", stats);
        },
        (site) => {
          if (site.domain && site.jobs_count) {
            sitesDiscovered[site.domain] = (sitesDiscovered[site.domain] || 0) + site.jobs_count;
            sitesDiscovered = sitesDiscovered;
          }
          dispatch("siteFound", site);  // FIX: was missing dispatch here too
        },
        (plan) => { dispatch("planUpdate", plan); },
        (ev) => { dispatch("siteActive", ev); },
        (ev) => { dispatch("agentEvent", ev); },
      );

      const persisted = await getSessionJobs(sessionId).catch(() => null);
      if (persisted?.jobs?.length) {
        jobs = persisted.jobs;
        jobsFound = persisted.count ?? persisted.jobs.length;
      }
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

<div class="space-y-5">

  <!-- ── Status bar ─────────────────────────────────── -->
  {#if status !== "idle"}
    <section class="card px-4 py-3 sm:px-5">
      <div class="flex flex-wrap items-center justify-between gap-3">
        <div class="flex items-center gap-3">
          {#if status === "searching"}
            <span class="pulse-dot"></span>
          {:else if status === "done"}
            <span style="color:var(--green);font-size:1.1em;">✓</span>
          {:else if status === "error"}
            <span style="color:var(--red);font-size:1.1em;">✕</span>
          {/if}
          <span class="text-sm font-medium text-text">{statusMsg}</span>
        </div>
        <div class="grid grid-cols-2 gap-2 sm:flex sm:items-center sm:gap-4">
          {#each [["Jobs", jobsFound], ["Pages", pagesCrawled], ["Sites", sitesCount], ["★ Quality", highQuality]] as [label, val]}
            <div class="surface min-w-[72px] px-3 py-2 text-center">
              <div class="font-mono text-base font-semibold text-text-hi">{val}</div>
              <div class="text-[10px] uppercase tracking-[0.18em] text-text-dim">{label}</div>
            </div>
          {/each}
        </div>
      </div>

      <!-- Live traces — real-time agent reasoning inline under status bar -->
      {#if status === "searching" && traces.length > 0}
        <div class="mt-3 max-h-[72px] overflow-hidden border-t border-border pt-3 font-mono text-[10px]">
          {#each traces.slice(-3) as t}
            <div class="overflow-hidden text-ellipsis whitespace-nowrap text-text-dim">
              {t.icon} {t.message}
            </div>
          {/each}
        </div>
      {/if}
    </section>
  {/if}

  <!-- Filter + sort bar -->
  {#if jobs.length > 0}
    <section class="card p-3 sm:p-4">
    <div class="flex flex-wrap items-center gap-2">
      <input
        bind:value={searchTerm}
        placeholder="Filter results…"
        class="input w-full font-mono text-xs sm:w-48"
      />

      <!-- FIX: Remote filter shows active state when auto-set from search -->
      <select
        bind:value={filterRemote}
        class="input min-w-[120px] px-3 py-2 font-mono text-xs {filterRemote !== 'all' ? 'text-text-hi border-border-hi' : 'text-text-dim'}"
      >
        <option value="all">All modes</option>
        <option value="remote">Remote</option>
        <option value="hybrid">Hybrid</option>
        <option value="on-site">On-site</option>
      </select>

      <!-- FIX: Level filter shows active state when auto-set from search -->
      <select
        bind:value={filterLevel}
        class="input min-w-[120px] px-3 py-2 font-mono text-xs {filterLevel !== 'all' ? 'text-text-hi border-border-hi' : 'text-text-dim'}"
      >
        <option value="all">All levels</option>
        <option value="intern">Intern</option>
        <option value="junior">Junior</option>
        <option value="mid">Mid</option>
        <option value="mid-senior">Mid-Senior</option>
        <option value="senior">Senior</option>
      </select>

      <!-- Result count badge -->
      <span class="font-mono text-[10px] whitespace-nowrap text-text-dim">
        {filteredJobs.length} / {jobs.length} shown
        {#if filterRemote !== "all" || filterLevel !== "all" || searchTerm}
          — <button
            on:click={() => { filterRemote="all"; filterLevel="all"; searchTerm=""; }}
            class="border-none bg-transparent text-text-lo underline-offset-2 hover:text-text-hi hover:underline"
          >clear filters</button>
        {/if}
      </span>

      <div class="ml-auto flex gap-1">
        {#each [["live","Live"],["score","Relevance"],["recent","Recent"],["company","Company"]] as [val, label]}
          <button
            on:click={() => sortBy = val}
            class="rounded-full border px-3 py-1.5 text-xs font-mono transition-colors {
              sortBy === val
                ? 'border-border-hi bg-[var(--surface-3)] text-text-hi'
                : 'border-border text-text-dim hover:border-border-hi'
            }"
          >
            {label}
          </button>
        {/each}
      </div>
    </div>
    </section>
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
    <div class="card px-6 py-16 text-center">
      <div class="text-4xl text-text-dim">⌖</div>
      <p class="mt-4 text-2xl font-semibold tracking-wide text-text-hi">Start searching</p>
      <p class="mx-auto mt-2 max-w-md font-mono text-xs text-text-dim">
        Enter a query above. JobRadar will search across job boards, ATS pages,
        and company career sites — ranking by fit, not just keywords.
      </p>
    </div>
  {:else if status === "searching" && jobs.length === 0}
    <div class="space-y-3 pt-2">
      {#each Array(4) as _, i}
        <div
          class="card animate-pulse p-4"
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
    <div class="card py-14 text-center">
      <p class="text-xl text-text-dim">No results match current filters</p>
      <button
        on:click={() => { filterRemote="all"; filterLevel="all"; searchTerm=""; }}
        class="mt-2 font-mono text-xs text-text-lo hover:text-text-hi hover:underline"
      >
        Clear filters
      </button>
    </div>
  {/if}

</div>
