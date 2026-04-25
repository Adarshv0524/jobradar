<script lang="ts">
  export let open = false;
  export let filename = "";
  export let extracted: {
    skills: string[];
    experience_level: string;
    summary: string;
    raw_length?: number;
    experience?: {
      level?: string;
      confidence?: string;
      years_estimate?: number | null;
      internship_months?: number | null;
      graduation_year?: number | null;
      signals?: string[];
    };
    sections?: Record<string, string>;
    projects?: string[];
    keywords?: string[];
    extraction?: { method?: string; warnings?: string[]; page_char_counts?: number[] };
  } | null = null;

  export let onClose: () => void = () => {};

  function close() {
    open = false;
    onClose();
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === "Escape") close();
    if (e.key === "Enter" || e.key === " ") close();
  }

  function onOverlayClick(e: MouseEvent) {
    if (e.currentTarget === e.target) close();
  }

  $: warnList = extracted?.extraction?.warnings ?? [];
  $: pageCounts = extracted?.extraction?.page_char_counts ?? [];
  $: exp = extracted?.experience ?? {};
  $: expSignals = (exp?.signals ?? []) as string[];
  $: sections = extracted?.sections ?? {};
  $: projects = extracted?.projects ?? [];
  $: keywords = extracted?.keywords ?? [];
</script>

{#if open}
  <div
    class="neo-overlay"
    role="button"
    aria-label="Close resume insights"
    tabindex="0"
    on:keydown={onKeydown}
    on:click={onOverlayClick}
  >
    <div
      class="neo-modal neo-card"
      role="dialog"
      aria-modal="true"
      aria-label="Resume extraction details"
    >
      <div class="flex items-start justify-between gap-4 mb-4">
        <div class="min-w-0">
          <div class="text-[12px] font-semibold tracking-wider text-text-hi">RESUME INSIGHTS</div>
          <div class="font-mono text-[11px] text-text-dim truncate">{filename || "resume"}</div>
        </div>
        <button class="neo-btn btn-sm" on:click={() => close()} aria-label="Close">✕</button>
      </div>

      {#if !extracted}
        <div class="text-sm text-text-dim">No resume data loaded.</div>
      {:else}
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div class="neo-inset p-4">
            <div class="text-[10px] font-semibold tracking-wider text-text-dim mb-2">SUMMARY</div>
            <div class="text-[12px] text-text-lo whitespace-pre-wrap break-words">
              {extracted.summary || "—"}
            </div>
          </div>

          <div class="neo-inset p-4">
            <div class="text-[10px] font-semibold tracking-wider text-text-dim mb-2">EXPERIENCE INFERENCE</div>
            <div class="grid grid-cols-2 gap-2">
              <div class="surface px-2 py-2">
                <div class="text-[10px] text-text-dim">Level</div>
                <div class="text-[12px] text-text-hi font-mono">{(exp.level || extracted.experience_level || "unknown")}</div>
              </div>
              <div class="surface px-2 py-2">
                <div class="text-[10px] text-text-dim">Confidence</div>
                <div class="text-[12px] text-text-hi font-mono">{(exp.confidence || "low")}</div>
              </div>
              <div class="surface px-2 py-2">
                <div class="text-[10px] text-text-dim">Grad year</div>
                <div class="text-[12px] text-text-hi font-mono">{exp.graduation_year ?? "—"}</div>
              </div>
              <div class="surface px-2 py-2">
                <div class="text-[10px] text-text-dim">Intern months</div>
                <div class="text-[12px] text-text-hi font-mono">{exp.internship_months ?? "—"}</div>
              </div>
              <div class="surface px-2 py-2 col-span-2">
                <div class="text-[10px] text-text-dim">Years estimate</div>
                <div class="text-[12px] text-text-hi font-mono">{exp.years_estimate ?? "—"}</div>
              </div>
            </div>

            {#if expSignals.length}
              <div class="mt-2 text-[10px] text-text-dim font-semibold tracking-wider">SIGNALS</div>
              <div class="mt-1 flex flex-wrap gap-1.5">
                {#each expSignals.slice(0, 14) as s (s)}
                  <span class="badge badge-neutral">{s}</span>
                {/each}
                {#if expSignals.length > 14}
                  <span class="badge badge-neutral">+{expSignals.length - 14}</span>
                {/if}
              </div>
            {/if}
          </div>
        </div>

        <div class="neo-inset mt-3 p-4">
          <div class="text-[10px] font-semibold tracking-wider text-text-dim mb-2">SKILLS ({extracted.skills?.length ?? 0})</div>
          {#if extracted.skills?.length}
            <div class="flex flex-wrap gap-1.5 max-h-40 overflow-auto pr-1">
              {#each extracted.skills as sk (sk)}
                <span class="skill-tag">{sk}</span>
              {/each}
            </div>
          {:else}
            <div class="text-[12px] text-text-dim">No skills detected.</div>
          {/if}
        </div>

        {#if projects.length || sections.projects}
          <div class="neo-inset mt-3 p-4">
            <div class="text-[10px] font-semibold tracking-wider text-text-dim mb-2">PROJECTS</div>
            {#if projects.length}
              <ul class="text-[12px] text-text-lo list-disc pl-5 space-y-1">
                {#each projects as p (p)}
                  <li class="break-words">{p}</li>
                {/each}
              </ul>
            {:else}
              <div class="text-[12px] text-text-lo whitespace-pre-wrap break-words">
                {sections.projects}
              </div>
            {/if}
          </div>
        {/if}

        {#if keywords.length}
          <div class="neo-inset mt-3 p-4">
            <div class="text-[10px] font-semibold tracking-wider text-text-dim mb-2">KEYWORDS</div>
            <div class="flex flex-wrap gap-1.5">
              {#each keywords as k (k)}
                <span class="badge badge-neutral">{k}</span>
              {/each}
            </div>
          </div>
        {/if}

        <div class="neo-inset mt-3 p-4">
          <div class="text-[10px] font-semibold tracking-wider text-text-dim mb-2">EXTRACTION DIAGNOSTICS</div>
          <div class="grid grid-cols-2 gap-2">
            <div class="surface px-2 py-2">
              <div class="text-[10px] text-text-dim">Method</div>
              <div class="text-[12px] text-text-hi font-mono">{extracted.extraction?.method ?? "—"}</div>
            </div>
            <div class="surface px-2 py-2">
              <div class="text-[10px] text-text-dim">Chars</div>
              <div class="text-[12px] text-text-hi font-mono">{extracted.raw_length ?? "—"}</div>
            </div>
          </div>

          {#if warnList.length}
            <div class="mt-2 text-[11px]" style="color:var(--red);">
              {#each warnList as w (w)}
                <div class="font-mono">⚠ {w}</div>
              {/each}
            </div>
          {/if}

          {#if pageCounts.length}
            <div class="mt-2 text-[11px] text-text-dim font-mono">
              pages: {pageCounts.map((n, i) => `${i + 1}:${n}`).join("  ")}
            </div>
          {/if}
        </div>

        <div class="flex justify-end gap-2 mt-4">
          <button class="neo-btn" on:click={() => close()}>Close</button>
        </div>
      {/if}
    </div>
  </div>
{/if}
