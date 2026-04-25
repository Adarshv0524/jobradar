<script lang="ts">
  import { controlSearch, getAgentEvents, type AgentEventPayload, type SSEEvent } from "../lib/api";
  import { kindClass } from "../lib/theme";

  export let isSearching = false;
  export let className: string = "";

  let sessionId = "";
  let controlState: "running" | "paused" | "stopped" | "unknown" = "unknown";

  let stats: {
    jobs_found?: number;
    pages_crawled?: number;
    tokens_total?: number;
    high_quality?: number;
    sites_count?: number;
    llm_enabled?: boolean;
    model?: string | null;
    queries_used?: number;
  } = {};

  type ActionCard = AgentEventPayload & { _expanded?: boolean };
  let actions: ActionCard[] = [];
  let actionIds = new Set<string>();

  let focus: "llm" | "all" = "llm";

  function isLLMAction(a: AgentEventPayload): boolean {
    const t = (a.tool?.name || "").toLowerCase();
    if (a.kind === "error" || a.kind === "wait") return true;
    if (t.startsWith("llm:")) return true;
    if (["web_search", "select_urls", "expand_urls", "crawl_url", "evaluate_and_plan"].includes(t)) return true;
    if (a.kind === "decision") return true;
    if (a.kind === "stage" && ["init", "plan", "search", "evaluate", "watch_wait", "done"].includes(a.stage)) return true;
    return false;
  }

  $: visibleActions = (focus === "llm" ? actions.filter(isLLMAction) : actions).slice(0, 160);
  $: hiddenCount = Math.max(0, actions.length - visibleActions.length);
  $: reasoningActions = visibleActions.filter(a => a.kind !== "tool_call" && a.kind !== "tool_result");
  $: toolActions = visibleActions.filter(a => a.kind === "tool_call" || a.kind === "tool_result");

  $: brainPlan =
    actions.find(a => a.tool?.name === "llm:generate_search_queries")
    || actions.find(a => (a.tool?.name || "").includes("generate_queries"))
    || null;

  $: brainEval =
    actions.find(a => a.tool?.name === "evaluate_and_plan")
    || null;

  $: now =
    actions.find(a => a.kind === "tool_call" && a.tool?.status === "running")
    || actions.find(a => a.kind === "stage")
    || null;

  export async function setSessionId(sid: string) {
    sessionId = sid || "";
    actions = [];
    actionIds = new Set();
    controlState = "running";
    if (!sessionId) return;
    try {
      const r = await getAgentEvents(sessionId, 0, 300);
      const payloads: AgentEventPayload[] = (r.events || []).map((e: any) => e.payload).filter(Boolean);
      payloads.sort((a, b) => (b.ts || 0) - (a.ts || 0));
      actions = payloads.map(p => ({ ...p, _expanded: false }));
      actionIds = new Set(actions.map(a => a.id));
    } catch {
      // ignore
    }
  }

  export function addAgentEvent(ev: SSEEvent) {
    const payload = ev?.data as any as AgentEventPayload;
    if (!payload?.id) return;
    if (actionIds.has(payload.id)) return;
    actionIds.add(payload.id);
    actions = [{ ...payload, _expanded: false }, ...actions].slice(0, 300);

    if (payload.kind === "wait" && payload.title === "Paused") controlState = "paused";
    if (payload.kind === "error" && payload.title === "Stopped") controlState = "stopped";
    if (payload.kind === "stage" && payload.stage === "done") controlState = "stopped";
  }

  export function handleStats(data: any) {
    stats = { ...stats, ...data };
  }

  function fmtTime(ts: number): string {
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch { return ""; }
  }

  function sentence(text: string | undefined, fallback: string): string {
    const raw = (text || fallback).trim();
    if (!raw) return fallback;
    const normalized = raw.charAt(0).toUpperCase() + raw.slice(1);
    return /[.!?]$/.test(normalized) ? normalized : `${normalized}.`;
  }

  function reasoningText(a: ActionCard): string {
    if (a.title === "Meta-agent reasoning") {
      return sentence(a.detail, "The meta-agent is choosing the next best move.");
    }
    if (a.kind === "decision") {
      return sentence(a.detail, `The agent made a ${a.stage} decision.`);
    }
    if (a.kind === "stage") {
      return sentence(a.detail, `${a.title} is now in progress.`);
    }
    if (a.kind === "wait") {
      return sentence(a.detail, "The agent is waiting.");
    }
    if (a.kind === "error") {
      return sentence(a.detail, "The agent hit an error.");
    }
    if (a.kind === "metric") {
      return sentence(a.detail, "The agent reported a status update.");
    }
    return sentence(a.detail, a.title);
  }

  function toggleExpanded(id: string) {
    actions = actions.map(a => a.id === id ? { ...a, _expanded: !a._expanded } : a);
  }

  async function sendControl(action: "pause" | "resume" | "stop") {
    if (!sessionId) return;
    try {
      await controlSearch(sessionId, action);
      controlState = action === "pause" ? "paused" : action === "resume" ? "running" : "stopped";
    } catch {
      // ignore
    }
  }
</script>

<aside class={"card p-4 sm:p-5 " + className}>
  <div class="mb-4 flex items-center justify-between">
    <div class="flex items-center gap-2">
      <span class="text-[12px] font-semibold uppercase tracking-[0.24em] text-text-hi">Meta agent</span>
      {#if isSearching}
        <span class="rounded-full border border-border px-2 py-0.5 font-mono text-[10px]" style="color:var(--green);">LIVE</span>
      {/if}
    </div>
    <div class="flex gap-1">
      <button class="btn btn-ghost btn-sm" on:click={() => sendControl("pause")} disabled={!sessionId} title="Pause">⏸</button>
      <button class="btn btn-ghost btn-sm" on:click={() => sendControl("resume")} disabled={!sessionId} title="Resume">▶</button>
      <button class="btn btn-ghost btn-sm" on:click={() => sendControl("stop")} disabled={!sessionId} title="Stop">■</button>
    </div>
  </div>

  <div class="surface mb-4 px-3 py-3">
    <div class="flex items-center justify-between gap-2">
      <div class="text-[10px] text-text-dim font-mono truncate">
        {#if sessionId}
          session: <span class="text-text-lo">{sessionId.slice(0, 8)}…</span>
        {:else}
          session: <span class="text-text-lo">—</span>
        {/if}
        {" "}· state: <span class="text-text-lo">{controlState}</span>
      </div>
      <div class="flex gap-1">
        <button class="btn btn-ghost btn-sm {focus==='llm' ? 'border-border-hi text-text-hi bg-[var(--surface-3)]' : ''}" on:click={() => focus = "llm"}>LLM</button>
        <button class="btn btn-ghost btn-sm {focus==='all' ? 'border-border-hi text-text-hi bg-[var(--surface-3)]' : ''}" on:click={() => focus = "all"}>All</button>
      </div>
    </div>
    {#if focus === "llm" && hiddenCount > 0}
      <div class="text-[10px] text-text-dim font-mono mt-1">hidden noisy events: {hiddenCount}</div>
    {/if}
  </div>

  <div class="mb-4 grid grid-cols-2 gap-2">
    <div class="surface px-2 py-2">
      <div class="text-[10px] text-text-dim">Jobs</div>
      <div class="text-[13px] font-semibold text-text-hi font-mono">{stats.jobs_found ?? 0}</div>
    </div>
    <div class="surface px-2 py-2">
      <div class="text-[10px] text-text-dim">Pages</div>
      <div class="text-[13px] font-semibold text-text-hi font-mono">{stats.pages_crawled ?? 0}</div>
    </div>
    <div class="surface px-2 py-2">
      <div class="text-[10px] text-text-dim">High quality</div>
      <div class="text-[13px] font-semibold text-text-hi font-mono">{stats.high_quality ?? 0}</div>
    </div>
    <div class="surface px-2 py-2">
      <div class="text-[10px] text-text-dim">Tokens</div>
      <div class="text-[13px] font-semibold text-text-hi font-mono">{stats.tokens_total ?? 0}</div>
    </div>
  </div>

  <div class="surface mb-4 px-3 py-3">
    <div class="mb-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-text-dim">Brain</div>
    {#if now}
      <div class="text-[11px] text-text-lo">
        <span class="font-mono text-text-dim">{fmtTime(now.ts)}</span>
        {" "}· <span class="font-mono {kindClass(now.kind)}">[{now.stage}:{now.kind}]</span>
        {" "}<span class="text-text">{now.title}</span>
      </div>
      {#if now.detail}
        <div class="text-[10px] text-text-dim font-mono break-words mt-1">{now.detail}</div>
      {/if}
    {:else}
      <div class="text-[11px] text-text-dim">No active action yet</div>
    {/if}
    {#if brainPlan?.detail}
      <div class="divider my-2"></div>
      <div class="mb-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-text-dim">Plan</div>
      <div class="text-[10px] text-text-lo font-mono break-words">{brainPlan.detail}</div>
    {/if}
    {#if brainEval?.detail}
      <div class="divider my-2"></div>
      <div class="mb-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-text-dim">Evaluation</div>
      <div class="text-[10px] text-text-lo font-mono break-words">{brainEval.detail}</div>
    {/if}
  </div>

  <div class="mb-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-text-dim">Reasoning</div>
  <div class="mb-4 flex max-h-[280px] flex-col gap-1.5 overflow-y-auto">
    {#if reasoningActions.length === 0}
      <p class="py-4 text-center text-[12px] text-text-dim">
        {isSearching ? "Waiting for readable reasoning…" : "Start a search to see the agent's reasoning"}
      </p>
    {:else}
      {#each reasoningActions as a (a.id)}
        <div class="surface px-3 py-2">
          <div class="flex items-start gap-2">
            <div class="mt-0.5 text-[10px] font-mono text-text-dim">{fmtTime(a.ts)}</div>
            <div class="min-w-0 flex-1">
              <div class="text-[10px] font-mono uppercase tracking-[0.16em] text-text-dim">{a.stage}</div>
              <div class="mt-1 text-[12px] leading-5 text-text">{reasoningText(a)}</div>
            </div>
          </div>
        </div>
      {/each}
    {/if}
  </div>

  <div class="mb-2 text-[10px] font-semibold uppercase tracking-[0.2em] text-text-dim">Tool Calls</div>
  <div class="flex max-h-[320px] flex-col gap-1.5 overflow-y-auto">
    {#if toolActions.length === 0}
      <p class="text-[12px] text-text-dim text-center py-6">
        {isSearching ? "Waiting for tool activity…" : "Start a search to see tool calls"}
      </p>
    {:else}
      {#each toolActions as a (a.id)}
        <button
          class="surface w-full px-3 py-2 text-left transition-colors hover:bg-[var(--surface-2)]"
          on:click={() => toggleExpanded(a.id)}
        >
          <div class="flex items-start gap-2">
            <div class="text-[10px] font-mono text-text-dim mt-0.5">{fmtTime(a.ts)}</div>
            <div class="min-w-0 flex-1">
              <div class="text-[11px] text-text-lo break-words">
                <span class="font-mono {kindClass(a.kind)}">[{a.kind}]</span>
                {" "}<span class="text-text">{a.tool?.name || a.title}</span>
                <span class="text-[10px] text-text-dim font-mono ml-2">{a._expanded ? "▾" : "▸"}</span>
              </div>
              <div class="mt-1 text-[10px] font-mono text-text-dim">
                stage: {a.stage}
                {#if a.tool?.status} · status: {a.tool.status}{/if}
                {#if a.tool?.duration_ms} · {a.tool.duration_ms}ms{/if}
              </div>
              {#if a.detail}
                <div class="text-[10px] text-text-dim font-mono break-words mt-1">{a.detail}</div>
              {/if}
              {#if a._expanded && a.tool}
                {#if a.tool.input_summary}
                  <div class="text-[10px] text-text-dim font-mono break-words">in: {a.tool.input_summary}</div>
                {/if}
                {#if a.tool.output_summary}
                  <div class="text-[10px] text-text-dim font-mono break-words">out: {a.tool.output_summary}</div>
                {/if}
              {/if}
            </div>
          </div>
        </button>
      {/each}
    {/if}
  </div>
</aside>
