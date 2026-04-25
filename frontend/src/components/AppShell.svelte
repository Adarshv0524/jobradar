<script lang="ts">
  import SearchForm   from "./SearchForm.svelte";
  import JobBoard     from "./JobBoard.svelte";
  import AgentControlCenter from "./AgentControlCenter.svelte";
  import type { SearchPayload } from "../lib/api";

  let boardRef: JobBoard;
  let sidebarRef: AgentControlCenter;
  let pendingPayload: SearchPayload | null = null;
  let isSearching = false;
  let sessionId = "";

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

<div class="grid grid-cols-1 items-start gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">

  <main class="min-w-0 space-y-5">
    <SearchForm on:search={handleSearch} />
    <JobBoard
      bind:this={boardRef}
      on:searchStart={handleSearchStart}
      on:searchDone={handleSearchDone}
      on:stats={(e) => sidebarRef?.handleStats(e.detail)}
      on:session={(e) => { sessionId = e.detail.sessionId; sidebarRef?.setSessionId?.(sessionId); }}
      on:agentEvent={(e) => sidebarRef?.addAgentEvent?.(e.detail)}
    />
  </main>

  <aside class="lg:sticky lg:top-24 lg:max-h-[calc(100vh-7rem)] lg:overflow-y-auto">
    <AgentControlCenter bind:this={sidebarRef} {isSearching} className="mb-4" />
  </aside>

</div>
