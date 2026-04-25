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

        <!-- FIX: Experience level badge in card header (was only in expanded footer) -->
        {#if expLabel}
          <span
            class="badge"
            style={expBadgeStyle}
          >{expLabel}</span>
        {/if}

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

      <!-- Match reason / warnings -->
      {#if !expanded}
        {#if locWarn}
          <p style="font-size:11px;color:var(--red);font-family:'JetBrains Mono',monospace;">
            ⚠ US-only remote — may not be accessible from your location
          </p>
        {:else if expWarn}
          <p style="font-size:11px;color:var(--accent);font-family:'JetBrains Mono',monospace;">
            ⚠ Experience level mismatch
          </p>
        {:else if job.match_reasons?.length}
          <p style="font-size:11px;color:var(--green);font-family:'JetBrains Mono',monospace;">
            ✓ {job.match_reasons[0]}
          </p>
        {/if}
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
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:12px;">
          {#each Object.entries(job.score_breakdown) as [key, val]}
            {@const n = Number(val)}
            <!-- FIX: show exp_mult and loc_mult with distinct colors -->
            {@const isMult = key === "exp_mult" || key === "loc_mult"}
            <div class="surface" style="padding:8px;text-align:center;">
              <p style="font-size:10px;color:var(--text-dim);letter-spacing:0.06em;text-transform:uppercase;margin:0 0 2px;">{key}</p>
              <p class="mono" style="font-size:13px;font-weight:600;margin:0;
                color:{isMult
                  ? (n < 0.5 ? 'var(--red)' : n < 0.85 ? 'var(--accent)' : 'var(--green)')
                  : (n>0.5?'var(--green)':n>0.25?'var(--accent)':'var(--text-dim)')}">
                {isMult ? (n < 1 ? `×${n.toFixed(2)}` : "✓") : Math.round(n*100) + "%"}
              </p>
            </div>
          {/each}
        </div>
      {/if}

      <!-- Match / reject reasons -->
      {#if job.reject_reasons?.length}
        <p style="font-size:10px;font-weight:600;color:var(--text-dim);letter-spacing:0.07em;margin-bottom:6px;">WHY IT'S PENALISED</p>
        <div style="margin-bottom:12px;">
          {#each job.reject_reasons as r}
            <p style="font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--red);margin:2px 0;">⚠ {r}</p>
          {/each}
        </div>
      {/if}

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