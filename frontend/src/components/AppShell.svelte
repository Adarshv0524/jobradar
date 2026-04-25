<script lang="ts">
  import SearchForm   from "./SearchForm.svelte";
  import JobBoard     from "./JobBoard.svelte";
  import SiteSidebar  from "./SiteSidebar.svelte";
  import type { SearchPayload } from "../lib/api";

  let boardRef: JobBoard;
  let sidebarRef: SiteSidebar;
  let pendingPayload: SearchPayload | null = null;
  let isSearching = false;

  function handleSearch(e: CustomEvent<SearchPayload>) {
    pendingPayload = e.detail;
  }

  function handleSearchStart() { isSearching = true; }
  function handleSearchDone()  { isSearching = false; }

  $: if (pendingPayload && boardRef) {
    boardRef.startNewSearch(pendingPayload);
    pendingPayload = null;
  }
</script>

<div style="display:flex;gap:20px;align-items:flex-start;">

  <!-- ── Left Sidebar ─────────────────────────────────────── -->
  <aside style="
    width: 280px;
    flex-shrink: 0;
    position: sticky;
    top: 72px;
    max-height: calc(100vh - 92px);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
  ">
    <SiteSidebar bind:this={sidebarRef} {isSearching} />
  </aside>

  <!-- ── Main Content ─────────────────────────────────────── -->
  <main style="flex:1;min-width:0;display:flex;flex-direction:column;gap:16px;">
    <SearchForm on:search={handleSearch} />
    <JobBoard
      bind:this={boardRef}
      on:searchStart={handleSearchStart}
      on:searchDone={handleSearchDone}
      on:planUpdate={(e) => sidebarRef?.handlePlanUpdate(e.detail)}
      on:trace={(e) => sidebarRef?.addTrace(e.detail)}
      on:siteFound={(e) => sidebarRef?.markSiteFound(e.detail)}
    />
  </main>

</div>