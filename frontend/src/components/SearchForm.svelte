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