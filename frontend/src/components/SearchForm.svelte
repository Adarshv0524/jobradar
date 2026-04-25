<script lang="ts">
  import { createEventDispatcher } from "svelte";
  import { uploadResume, type SearchPayload } from "../lib/api";
  import ResumeInsightsModal from "./ResumeInsightsModal.svelte";

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
  let watchMode        = false;
  let watchIntervalMin = "15";
  let watchMaxCycles   = "";
  let uploading        = false;
  let uploadMsg        = "";
  let uploadedName     = "";
  let resumeExtracted: Awaited<ReturnType<typeof uploadResume>> | null = null;
  let resumeModalOpen  = false;
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
    uploadedName = file.name;
    try {
      const r = await uploadResume(file);
      resumeExtracted = r;
      skills  = r.skills.join(", ");
      summary = r.summary;
      if (r.experience_level && r.experience_level !== "unknown") experienceLevel = r.experience_level;
      uploadMsg = `✓ ${r.skills.length} skills extracted`;
      resumeModalOpen = true;
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
      watch_mode:         watchMode,
      watch_interval_sec: Math.max(60, parseInt(watchIntervalMin || "15") * 60),
      watch_max_cycles:   watchMaxCycles ? parseInt(watchMaxCycles) : 0,
    });
  }
</script>

<section class="card p-4 sm:p-6">
  <div class="mb-5 flex flex-col gap-2 border-b border-border pb-4 sm:flex-row sm:items-end sm:justify-between">
    <div>
      <p class="text-[11px] font-semibold uppercase tracking-[0.24em] text-text-dim">Search setup</p>
      <h1 class="mt-1 text-xl font-semibold text-text-hi sm:text-2xl">Find cleaner matches faster</h1>
    </div>
    <p class="max-w-xl text-sm text-text-lo">Use a focused query, keep filters narrow, and let the board stay quiet and readable while results stream in.</p>
  </div>

  <div class="mb-4 flex flex-col gap-2 lg:flex-row">
    <input
      class="input grow"
      bind:value={query}
      placeholder="e.g. Senior Python Engineer, Data Scientist, DevOps Lead…"
      on:keydown={e => e.key === "Enter" && handleSubmit()}
    />
    <label class="btn btn-ghost lg:min-w-[150px]" title="Upload resume to auto-fill skills and summary">
      {uploading ? "Uploading…" : "Upload resume"}
      <input
        type="file" accept=".pdf,.txt" style="display:none;"
        on:change={handleResumeUpload} disabled={uploading}
      />
    </label>
    <button class="btn btn-primary lg:min-w-[120px]" on:click={handleSubmit} disabled={!query.trim()}>
      Search
    </button>
  </div>

  {#if uploadMsg}
    <div class="surface mb-4 flex items-center gap-2 px-3 py-2.5">
      <span class="font-mono text-[11px] text-text-lo truncate min-w-0 flex-1">
        {uploadedName ? uploadedName : "Resume"} — {uploadMsg}
      </span>
      {#if resumeExtracted}
        <button
          class="btn btn-ghost btn-sm"
          on:click={() => resumeModalOpen = true}
        >
          View
        </button>
      {/if}
      <button class="btn btn-ghost btn-sm" on:click={() => { uploadMsg=""; uploadedName="" }}>
        Dismiss
      </button>
    </div>
  {/if}

  <div class="grid gap-3 md:grid-cols-2 xl:grid-cols-[1.1fr_1.1fr_1fr_auto]">

    <select class="input pr-7" bind:value={remotePreference}>
      <option value="any">Any work type</option>
      <option value="remote">Remote only</option>
      <option value="hybrid">Hybrid</option>
      <option value="on-site">On-site</option>
    </select>

    <select class="input pr-7" bind:value={experienceLevel}>
      <option value="">Any level</option>
      <option value="junior">Junior</option>
      <option value="mid">Mid-level</option>
      <option value="senior">Senior</option>
      <option value="lead">Lead / Staff</option>
    </select>

    <input class="input" bind:value={location} placeholder="Location (optional)" />

    <button
      class="btn btn-ghost"
      on:click={() => showAdvanced = !showAdvanced}
    >
      {showAdvanced ? "▲" : "▼"} Advanced
    </button>
  </div>

  <!-- Advanced panel -->
  {#if showAdvanced}
    <div class="fade-in mt-5 border-t border-border pt-5">
      <div class="mb-4 grid grid-cols-1 gap-3 md:grid-cols-2">

        <div>
          <label class="label" for="minSalary">Min Salary (K USD)</label>
          <input id="minSalary" class="input" type="number" bind:value={minSalary} placeholder="e.g. 80" />
        </div>

        <div>
          <label class="label" for="skills">Skills (comma-separated)</label>
          <input id="skills" class="input" bind:value={skills} placeholder="python, react, aws…" />
        </div>
      </div>

      <div class="mb-4">
        <label class="label" for="summary">Profile Summary (improves ranking)</label>
        <textarea
          id="summary" class="input" bind:value={summary} rows="2"
          placeholder="Brief description of your background and goals…"
          style="resize:vertical;"
        ></textarea>
      </div>

      <div class="mb-4">
        <div class="label">Watch mode (keeps scanning)</div>
        <div class="flex flex-wrap items-center gap-2">
          <button
            class="btn btn-ghost btn-sm"
            on:click={() => watchMode = !watchMode}
          >
            {watchMode ? "✓ Enabled" : "Disabled"}
          </button>
          <input
            class="input w-28"
            type="number"
            min="1"
            bind:value={watchIntervalMin}
            disabled={!watchMode}
            placeholder="15"
          />
          <span class="text-[11px] text-text-dim font-mono">min interval</span>
          <input
            class="input w-28"
            type="number"
            min="0"
            bind:value={watchMaxCycles}
            disabled={!watchMode}
            placeholder="0"
          />
          <span class="text-[11px] text-text-dim font-mono">max cycles (0=default)</span>
        </div>
      </div>

      <div>
        <div class="label">Exclude</div>
        <div class="flex flex-wrap gap-2">
          {#each NEGATIVE_OPTIONS as opt}
            <button
              on:click={() => toggleNeg(opt.key)}
              class="btn btn-ghost btn-sm"
              class:neg-selected={negatives.includes(opt.key)}
            >{opt.label}</button>
          {/each}
        </div>
      </div>
    </div>
  {/if}
</section>

<ResumeInsightsModal
  open={resumeModalOpen}
  filename={uploadedName}
  extracted={resumeExtracted}
  onClose={() => resumeModalOpen = false}
/>
