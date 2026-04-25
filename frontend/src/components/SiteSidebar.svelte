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