<script lang="ts">
  import { controlSearch, getAgentEvents, type AgentEventPayload, type SSEEvent } from "../lib/api";
  import { kindClass, statusClass, statusIcon, traceClass } from "../lib/theme";
  export let isSearching = false;
  export let className: string = "";

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
  let agentEvents: AgentEventPayload[] = [];
  let agentEventIds: Set<string> = new Set();
  let expandedEventIds: Set<string> = new Set();
  let sessionId = "";
  let activeTab: "timeline" | "sites" | "reasoning" = "timeline";
  let timelineFocus: "llm" | "all" | "crawl" = "llm";
  let totalJobs = 0;
  let siteTally: Record<string, number> = {};
  let controlState: "running" | "paused" | "stopped" | "unknown" = "unknown";

  let stats: {
    jobs_found?: number;
    pages_crawled?: number;
    tokens_total?: number;
    high_quality?: number;
    sites_count?: number;
    llm_enabled?: boolean;
    llm_guided?: boolean;
    model?: string | null;
    queries_used?: number;
  } = {};

  let activeSite: { domain?: string; url?: string; site_label?: string; site_key?: string } = {};

  export async function setSessionId(sid: string) {
    sessionId = sid || "";
    agentEvents = [];
    agentEventIds = new Set();
    expandedEventIds = new Set();
    controlState = "running";
    if (!sessionId) return;
    try {
      const r = await getAgentEvents(sessionId, 0, 250);
      const payloads: AgentEventPayload[] = (r.events || [])
        .map((e: any) => e.payload)
        .filter(Boolean);
      payloads.sort((a, b) => (a.ts || 0) - (b.ts || 0));
      agentEvents = payloads.reverse().slice(0, 250);
      agentEventIds = new Set(agentEvents.map(e => e.id));
    } catch {
      // ignore
    }
  }

  export function addAgentEvent(ev: SSEEvent) {
    const payload = ev?.data as any as AgentEventPayload;
    if (!payload?.id) return;
    if (agentEventIds.has(payload.id)) return;
    agentEventIds.add(payload.id);
    agentEvents = [payload, ...agentEvents].slice(0, 250);
    if (payload.kind === "wait" && payload.title === "Paused") controlState = "paused";
    if (payload.kind === "error" && payload.title === "Stopped") controlState = "stopped";
    if (payload.kind === "stage" && payload.stage === "done") controlState = "stopped";
  }

  export function handlePlanUpdate(data: SitePlan) {
    sites = new Map(sites.set(data.site_key, data));
    if (data.status === "done" && data.jobs_found > 0) {
      totalJobs += data.jobs_found;
    }
  }

  export function addTrace(trace: TraceEntry) {
    traces = [trace, ...traces].slice(0, 60);
  }

  export function markSiteFound(data: { domain?: string; jobs_count?: number }) {
    if (!data?.domain || !data?.jobs_count) return;
    siteTally[data.domain] = (siteTally[data.domain] || 0) + data.jobs_count;
    siteTally = siteTally;
  }

  export function handleStats(data: any) {
    stats = { ...stats, ...data };
  }

  export function handleSiteActive(ev: any) {
    activeSite = {
      domain: ev?.domain,
      url: ev?.url,
      site_label: ev?.site_label,
      site_key: ev?.site_key,
    };
  }

  $: runningCount = [...sites.values()].filter(s => s.status === "running").length;
  $: doneCount    = [...sites.values()].filter(s => s.status === "done").length;
  $: failedCount  = [...sites.values()].filter(s => s.status === "failed").length;
  $: topSites = Object.entries(siteTally).sort((a, b) => b[1] - a[1]).slice(0, 6);

  // status, trace and kind mappings moved to frontend/src/lib/theme.ts

  function fmtTime(ts: number): string {
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch { return ""; }
  }

  function toggleExpanded(id: string) {
    if (!id) return;
    if (expandedEventIds.has(id)) {
      expandedEventIds.delete(id);
    } else {
      expandedEventIds.add(id);
    }
    expandedEventIds = new Set(expandedEventIds);
  }

  function isLLMRelevant(ev: AgentEventPayload): boolean {
    const toolName = (ev.tool?.name || "").toLowerCase();
    if (ev.kind === "error" || ev.kind === "wait") return true;
    if (toolName.startsWith("llm:")) return true;
    if (["web_search", "prioritize_urls", "evaluate_and_plan", "propose_company_sites"].includes(toolName)) return true;
    if (ev.kind === "decision" && (ev.stage === "plan" || ev.stage === "evaluate")) return true;
    if (ev.kind === "stage" && ["init", "plan", "search", "evaluate", "done", "watch_wait"].includes(ev.stage)) return true;
    return false;
  }

  function isCrawl(ev: AgentEventPayload): boolean {
    const toolName = (ev.tool?.name || "").toLowerCase();
    if (ev.stage === "crawl") return true;
    if (toolName.startsWith("crawl:")) return true;
    if (toolName === "crawl_url" || toolName === "rendered_fetch") return true;
    return false;
  }

  $: llmEvents = agentEvents.filter(isLLMRelevant);
  $: crawlEvents = agentEvents.filter(isCrawl);
  $: hiddenLLMCount = Math.max(0, agentEvents.length - llmEvents.length);
  $: displayEvents =
    timelineFocus === "llm" ? llmEvents :
    timelineFocus === "crawl" ? crawlEvents :
    agentEvents;

  $: lastPlan = agentEvents.find(e => (e.tool?.name || "") === "llm:generate_search_queries")
    ?? agentEvents.find(e => (e.tool?.name || "").includes("generate_queries"));

  async function sendControl(action: "pause" | "resume" | "stop") {
    if (!sessionId) return;
    try {
      await controlSearch(sessionId, action);
      controlState = action === "pause" ? "paused" : action === "resume" ? "running" : "stopped";
      traces = [{
        level: "plan",
        icon: action === "pause" ? "⏸" : action === "resume" ? "▶" : "■",
        message: `control: ${action}`,
        ts: Date.now(),
      }, ...traces].slice(0, 60);
    } catch {
      traces = [{ level: "warn", icon: "⚠", message: "control failed", ts: Date.now() }, ...traces].slice(0, 60);
    }
  }
</script>

<!-- ── Header ─────────────────────────────────────── -->
<div class={"card p-3.5 " + className}>
  <div class="flex items-center justify-between mb-3">
    <span class="text-[12px] font-semibold text-text-hi tracking-wider">AGENT CONTROL</span>
    {#if isSearching}
      <div class="flex items-center gap-1.5">
        <div class="pulse-dot"></div>
        <span class="text-[10px] font-mono text-green tracking-wider">LIVE</span>
      </div>
    {/if}
  </div>

  <div class="flex items-center justify-between mb-3 gap-2">
    <div class="text-[10px] text-text-dim font-mono truncate">
      {#if sessionId}
        session: <span class="text-text-lo">{sessionId.slice(0, 8)}…</span>
      {:else}
        session: <span class="text-text-lo">—</span>
      {/if}
      {" "}· state: <span class="text-text-lo">{controlState}</span>
    </div>
    <div class="flex gap-1">
      <button
        class="btn btn-ghost"
        style="padding:4px 8px;font-size:11px;"
        on:click={() => sendControl("pause")}
        disabled={!sessionId}
        title="Pause"
      >⏸</button>
      <button
        class="btn btn-ghost"
        style="padding:4px 8px;font-size:11px;"
        on:click={() => sendControl("resume")}
        disabled={!sessionId}
        title="Resume"
      >▶</button>
      <button
        class="btn btn-ghost"
        style="padding:4px 8px;font-size:11px;"
        on:click={() => sendControl("stop")}
        disabled={!sessionId}
        title="Stop"
      >■</button>
    </div>
  </div>

  {#if isSearching && (activeSite.domain || activeSite.site_label)}
    <div class="neo-inset px-3 py-2 mb-3">
      <div class="text-[10px] text-text-dim font-semibold tracking-wider">NOW CRAWLING</div>
      <div class="text-[12px] text-text-hi truncate">
        {activeSite.site_label || activeSite.domain || "—"}
      </div>
      {#if activeSite.url}
        <div class="text-[10px] text-text-dim font-mono truncate">{activeSite.url}</div>
      {/if}
    </div>
  {/if}

  <!-- Stats row -->
  <div class="grid grid-cols-3 gap-2 mb-3.5">
    <div class="surface px-2 py-2 text-center">
      <div class="text-[18px] font-bold text-green font-mono">{doneCount}</div>
      <div class="text-[10px] text-text-dim mt-0.5">Done</div>
    </div>
    <div class="surface px-2 py-2 text-center">
      <div class="text-[18px] font-bold text-accent font-mono">{runningCount}</div>
      <div class="text-[10px] text-text-dim mt-0.5">Active</div>
    </div>
    <div class="surface px-2 py-2 text-center">
      <div class="text-[18px] font-bold text-red font-mono">{failedCount}</div>
      <div class="text-[10px] text-text-dim mt-0.5">Failed</div>
    </div>
  </div>

  <div class="grid grid-cols-2 gap-2 mb-3.5">
    <div class="surface px-2 py-2">
      <div class="text-[10px] text-text-dim">Jobs</div>
      <div class="text-[13px] font-semibold text-text-hi font-mono">{stats.jobs_found ?? totalJobs ?? 0}</div>
    </div>
    <div class="surface px-2 py-2">
      <div class="text-[10px] text-text-dim">High quality</div>
      <div class="text-[13px] font-semibold text-text-hi font-mono">{stats.high_quality ?? 0}</div>
    </div>
    <div class="surface px-2 py-2">
      <div class="text-[10px] text-text-dim">Pages</div>
      <div class="text-[13px] font-semibold text-text-hi font-mono">{stats.pages_crawled ?? 0}</div>
    </div>
    <div class="surface px-2 py-2">
      <div class="text-[10px] text-text-dim">Tokens</div>
      <div class="text-[13px] font-semibold text-text-hi font-mono">{stats.tokens_total ?? 0}</div>
    </div>
  </div>

  <div class="flex items-center justify-between mb-3">
    <div class="text-[10px] text-text-dim font-mono">
      {#if stats.llm_enabled}
        model: <span class="text-text-lo">{stats.model ?? "enabled"}</span>
      {:else}
        model: <span class="text-text-lo">heuristic</span>
      {/if}
    </div>
    {#if stats.queries_used !== undefined}
      <div class="text-[10px] text-text-dim font-mono">
        queries: <span class="text-text-lo">{stats.queries_used}</span>
      </div>
    {/if}
  </div>

  <!-- Tabs -->
  <div class="flex gap-1 mb-3">
    {#each [["timeline","Timeline"],["sites","Sites"],["reasoning","Reasoning"]] as [tab, label]}
      <button
        on:click={() => activeTab = tab}
        class="flex-1 px-2 py-1.5 rounded text-[11px] font-medium border transition-colors
          {activeTab===tab ? 'bg-elevated border-border-hi text-text' : 'bg-transparent border-transparent text-text-dim hover:border-border'}"
      >{label}</button>
    {/each}
  </div>

  <!-- ── Timeline tab ───────────────────────────────── -->
  {#if activeTab === "timeline"}
    <div class="mb-2 flex items-center justify-between gap-2">
      <div class="flex gap-1">
        {#each [["llm","LLM"],["all","All"],["crawl","Crawl"]] as [val, label]}
          <button
            class="px-2 py-1 rounded text-[10px] font-mono border transition-colors
              {timelineFocus===val ? 'bg-elevated border-border-hi text-text' : 'bg-transparent border-border text-text-dim hover:border-border-hi'}"
            on:click={() => timelineFocus = val}
          >{label}</button>
        {/each}
      </div>
      {#if timelineFocus === "llm" && hiddenLLMCount > 0}
        <div class="text-[10px] text-text-dim font-mono">hidden: {hiddenLLMCount}</div>
      {/if}
    </div>

    {#if timelineFocus === "llm" && lastPlan?.detail}
      <div class="surface px-2.5 py-2 mb-2">
        <div class="text-[10px] text-text-dim font-semibold tracking-wider mb-1">LLM PLAN</div>
        <div class="text-[10px] text-text-lo font-mono break-words">{lastPlan.detail}</div>
      </div>
    {/if}

    <div class="flex flex-col gap-1.5 max-h-[520px] overflow-y-auto">
      {#if displayEvents.length === 0}
        <p class="text-[12px] text-text-dim text-center py-5">
          {isSearching ? "Agent timeline will appear here…" : "Start a search to see the agent timeline"}
        </p>
      {:else}
        {#each displayEvents as ev (ev.id)}
          <button
            class="slide-in text-left w-full px-2 py-1.5 rounded hover:bg-elevated"
            on:click={() => toggleExpanded(ev.id)}
          >
            <div class="flex gap-2 items-start">
              <span class="text-[10px] flex-shrink-0 mt-1 font-mono text-text-dim">{fmtTime(ev.ts)}</span>
              <div class="min-w-0 flex-1">
                <div class="text-[11px] text-text-lo leading-5 break-words">
                  <span class="font-medium {kindClass(ev.kind)}">[{ev.stage}:{ev.kind}]</span>
                  {" "}<span class="text-text">{ev.title}</span>
                  <span class="text-[10px] text-text-dim font-mono ml-2">{expandedEventIds.has(ev.id) ? "▾" : "▸"}</span>
                </div>
                {#if ev.detail}
                  <div class="text-[10px] text-text-dim font-mono break-words">{ev.detail}</div>
                {/if}
                {#if ev.tool?.name}
                  <div class="text-[10px] text-text-dim font-mono break-words">
                    tool: <span class="text-text-lo">{ev.tool.name}</span>
                    {#if ev.tool.status} · <span class="text-text-lo">{ev.tool.status}</span>{/if}
                    {#if ev.tool.duration_ms} · <span class="text-text-lo">{ev.tool.duration_ms}ms</span>{/if}
                  </div>
                {/if}
                {#if expandedEventIds.has(ev.id) && ev.tool}
                  {#if ev.tool.input_summary}
                    <div class="text-[10px] text-text-dim font-mono break-words">in: {ev.tool.input_summary}</div>
                  {/if}
                  {#if ev.tool.output_summary}
                    <div class="text-[10px] text-text-dim font-mono break-words">out: {ev.tool.output_summary}</div>
                  {/if}
                {/if}
              </div>
            </div>
          </button>
        {/each}
      {/if}
    </div>

  <!-- ── Sites tab ──────────────────────────────────── -->
  {:else if activeTab === "sites"}
    <div class="flex flex-col gap-1 max-h-[420px] overflow-y-auto">
      {#if sites.size === 0}
        <p class="text-[12px] text-text-dim text-center py-5">
          {isSearching ? "Building crawl plan…" : "Start a search to see sites"}
        </p>
      {:else}
        {#each [...sites.values()].sort((a,b) => {
          const order = {running:0,done:1,pending:2,failed:3,skipped:4};
          return (order[a.status]??5) - (order[b.status]??5);
        }) as site (site.site_key)}
          <div
            class="slide-in flex items-center gap-2 px-2 py-1.5 rounded border transition-colors
              {site.status==='running' ? 'bg-amber/5 border-amber/15' : 'bg-transparent border-transparent hover:border-border'}"
          >
            <span class="text-[10px] w-3 flex-shrink-0 font-mono {statusClass(site.status)}">{statusIcon(site.status)}</span>
            <span class="flex-1 text-[11px] min-w-0 truncate {site.status==='pending' ? 'text-text-dim' : 'text-text'}">
              {site.site_label}
            </span>
            {#if site.status === "done" && site.jobs_found > 0}
              <span class="text-[10px] font-mono text-green flex-shrink-0">
                +{site.jobs_found}
              </span>
            {:else if site.status === "running"}
              <span class="text-[10px] font-mono text-accent flex-shrink-0">
                p.{site.pages_done}
              </span>
            {/if}
          </div>
        {/each}
      {/if}

      {#if topSites.length}
        <div class="divider my-3"></div>
        <div class="text-[10px] font-semibold text-text-dim tracking-wider">TOP SOURCES</div>
        <div class="mt-2 flex flex-col gap-1">
          {#each topSites as [domain, count] (domain)}
            <div class="flex items-center gap-2 px-2 py-1 rounded hover:bg-elevated">
              <span class="text-[11px] text-text-lo truncate min-w-0 flex-1">{domain}</span>
              <span class="text-[11px] font-mono text-green flex-shrink-0">{count}</span>
            </div>
          {/each}
        </div>
      {/if}
    </div>

  <!-- ── Reasoning tab ──────────────────────────────── -->
  {:else}
    <div class="flex flex-col gap-1.5 max-h-[420px] overflow-y-auto">
      {#if traces.length === 0}
        <p class="text-[12px] text-text-dim text-center py-5">
          {isSearching ? "Agent is thinking…" : "Agent reasoning will appear here"}
        </p>
      {:else}
        {#each traces as trace (trace.ts)}
          <div class="slide-in px-2 py-1.5 rounded hover:bg-elevated">
            <div class="flex gap-2 items-start">
              <span class="text-[12px] flex-shrink-0 mt-0.5">{trace.icon}</span>
                <p class="text-[11px] text-text-lo leading-5 m-0 break-words">
                <span class="font-medium {traceClass(trace.level)}">[{trace.level}]</span>
                {" "}{trace.message}
              </p>
            </div>
          </div>
        {/each}
      {/if}
    </div>
  {/if}
</div>
