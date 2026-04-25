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

  // FIX: Experience level badge — shown prominently in card header
  $: expLabel = {
    intern:       "Intern",
    fresher:      "Fresher",
    junior:       "Junior",
    mid:          "Mid",
    "mid-senior": "Mid-Sr",
    senior:       "Senior",
    lead:         "Lead",
    staff:        "Staff",
    principal:    "Principal",
  }[job.experience_level?.toLowerCase() ?? ""] ?? (job.experience_level ?? "");

  $: expBadgeStyle = (() => {
    const lvl = job.experience_level?.toLowerCase() ?? "";
    if (lvl === "intern" || lvl === "fresher") return "color:#a78bfa;border-color:rgba(167,139,250,0.3);background:rgba(167,139,250,0.08);";
    if (lvl === "junior")     return "color:var(--green);border-color:rgba(34,197,94,0.3);background:var(--green-lo);";
    if (lvl === "mid")        return "color:var(--blue);border-color:rgba(96,165,250,0.3);background:var(--blue-lo);";
    if (lvl === "mid-senior") return "color:var(--blue);border-color:rgba(96,165,250,0.3);background:var(--blue-lo);";
    if (lvl === "senior")     return "color:var(--accent);border-color:var(--accent-hi);background:var(--accent-lo);";
    if (["lead","staff","principal"].includes(lvl)) return "color:var(--red);border-color:rgba(248,113,113,0.3);background:var(--red-lo);";
    return "color:var(--text-dim);border-color:var(--border);background:var(--surface);";
  })();

  function fmt(d: string): string {
    if (!d) return "";
    try { return new Date(d).toLocaleDateString("en-US", { month: "short", day: "numeric" }); }
    catch { return d; }
  }

  $: sourceLabel =
    job.source_type === "jsonld"       ? "structured" :
    job.source_type === "detail_page"  ? "detail page" :
    job.source_type?.startsWith("api") ? "API" : "scraped";

  // FIX: show exp_mult and loc_mult warnings if they pulled the score down
  $: expWarn = (job.score_breakdown as any)?.exp_mult < 0.5;
  $: locWarn = (job.score_breakdown as any)?.loc_mult < 0.5;
</script>

<article class="card card-hover fade-in relative p-4 sm:p-5" style="animation-delay: {Math.min(index * 40, 400)}ms;">
  <div class="absolute left-0 right-0 top-0 h-px overflow-hidden rounded-t-[20px]">
    <div style="height:100%; width:{scorePercent}%; background:{scoreColor}; transition:width 0.7s;"></div>
  </div>

  <div class="flex flex-col gap-4 lg:flex-row lg:items-start">
    <div class="min-w-0 flex-1">
      <div class="mb-3 flex flex-wrap items-center gap-2">
        <span class="mono text-[10px] text-text-dim">#{String(index+1).padStart(2,"0")}</span>
        <span class="badge {remoteBadgeClass}">{remoteLabel}</span>
        {#if expLabel}
          <span class="badge" style={expBadgeStyle}>{expLabel}</span>
        {/if}
        {#if job.posted_date}
          <span class="mono text-[10px] text-text-dim">{fmt(job.posted_date)}</span>
        {/if}
        <span class="mono ml-auto text-[10px] uppercase tracking-[0.18em] text-text-dim">{sourceLabel}</span>
      </div>

      <h3 class="mb-1 truncate text-lg font-semibold leading-tight text-text-hi">{job.title || "Untitled Role"}</h3>

      <div class="mb-4 flex flex-wrap items-center gap-2 text-sm">
        <span class="font-medium text-text">{job.company || "Unknown"}</span>
        {#if job.location}
          <span class="text-text-dim">·</span>
          <span class="text-text-lo">{job.location}</span>
        {/if}
        {#if job.salary}
          <span class="text-text-dim">·</span>
          <span class="badge badge-accent">{job.salary}</span>
        {/if}
      </div>

      {#if job.skills?.length}
        <div class="mb-3 flex flex-wrap gap-2">
          {#each job.skills.slice(0,8) as skill}
            <span class="skill-tag">{skill}</span>
          {/each}
          {#if job.skills.length > 8}
            <span class="badge badge-neutral">+{job.skills.length - 8}</span>
          {/if}
        </div>
      {/if}

      {#if !expanded}
        {#if locWarn}
          <p class="font-mono text-[11px]" style="color:var(--red);">⚠ US-only remote — may not be accessible from your location</p>
        {:else if expWarn}
          <p class="font-mono text-[11px] text-text-lo">⚠ Experience level mismatch</p>
        {:else if job.match_reasons?.length}
          <p class="font-mono text-[11px]" style="color:var(--green);">✓ {job.match_reasons[0]}</p>
        {/if}
      {/if}
    </div>

    <div class="flex shrink-0 flex-row items-center justify-between gap-3 lg:flex-col lg:items-end">
      <div class="mono rounded-full border px-3 py-1.5 text-[13px] font-semibold" style="border-color:{scoreColor}40; background:{scoreColor}12; color:{scoreColor};">
        {scorePercent}%
      </div>

      <div class="flex gap-2">
        <button
          on:click={() => rate(1)}
          class="flex h-9 w-9 items-center justify-center rounded-xl border text-sm transition-colors"
          style="border-color:{myRating===1 ? 'var(--green)' : 'var(--border)'}; background:{myRating===1 ? 'var(--green-lo)' : 'transparent'}; color:{myRating===1 ? 'var(--green)' : 'var(--text-dim)'};"
          title="Good match"
        >↑</button>
        <button
          on:click={() => rate(-1)}
          class="flex h-9 w-9 items-center justify-center rounded-xl border text-sm transition-colors"
          style="border-color:{myRating===-1 ? 'var(--red)' : 'var(--border)'}; background:{myRating===-1 ? 'var(--red-lo)' : 'transparent'}; color:{myRating===-1 ? 'var(--red)' : 'var(--text-dim)'};"
          title="Not relevant"
        >↓</button>
      </div>

      <a
        href={job.apply_url} target="_blank" rel="noopener noreferrer"
        class="btn btn-ghost btn-sm font-mono uppercase tracking-[0.16em] no-underline"
      >APPLY →</a>
    </div>
  </div>

  <button
    on:click={() => expanded = !expanded}
    class="mt-3 border-none bg-transparent p-0 font-mono text-[11px] text-text-dim transition-colors hover:text-text-lo"
  >{expanded ? "▲ hide details" : "▼ show details"}</button>

  {#if expanded}
    <div class="fade-in mt-4 border-t border-border pt-4">
      {#if job.score_breakdown && Object.keys(job.score_breakdown).length}
        <p class="mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-dim">Signal breakdown</p>
        <div class="mb-4 grid grid-cols-2 gap-2 md:grid-cols-4">
          {#each Object.entries(job.score_breakdown) as [key, val]}
            {@const n = Number(val)}
            {@const isMult = key === "exp_mult" || key === "loc_mult"}
            <div class="surface px-3 py-2 text-center">
              <p class="mb-1 text-[10px] uppercase tracking-[0.14em] text-text-dim">{key}</p>
              <p class="mono text-[13px] font-semibold" style="
                color:{isMult
                  ? (n < 0.5 ? 'var(--red)' : n < 0.85 ? 'var(--accent)' : 'var(--green)')
                  : (n>0.5?'var(--green)':n>0.25?'var(--accent)':'var(--text-dim)')};
              ">
                {isMult ? (n < 1 ? `×${n.toFixed(2)}` : "✓") : Math.round(n*100) + "%"}
              </p>
            </div>
          {/each}
        </div>
      {/if}

      {#if job.reject_reasons?.length}
        <p class="mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-dim">Why it&apos;s penalised</p>
        <div class="mb-4 space-y-1">
          {#each job.reject_reasons as r}
            <p class="font-mono text-[11px]" style="color:var(--red);">⚠ {r}</p>
          {/each}
        </div>
      {/if}

      {#if job.match_reasons?.length}
        <p class="mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-dim">Why it matched</p>
        <div class="mb-4 space-y-1">
          {#each job.match_reasons as r}
            <p class="font-mono text-[11px]" style="color:var(--green);">✓ {r}</p>
          {/each}
        </div>
      {/if}

      {#if job.description}
        <p class="mb-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-dim">Description</p>
        <p class="mb-4 text-[12px] leading-6 text-text-lo">
          {job.description.slice(0,700)}{job.description.length>700?"…":""}
        </p>
      {/if}

      <div class="flex flex-wrap gap-3 font-mono text-[11px] text-text-dim">
      <span>src: {job.source_domain}</span>
        {#if job.experience_level}
          <span>· {job.experience_level}</span>
        {/if}
      </div>
    </div>
  {/if}
</article>
