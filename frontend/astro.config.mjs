import { defineConfig } from "astro/config";
import svelte           from "@astrojs/svelte";
import tailwind         from "@astrojs/tailwind";

export default defineConfig({
  integrations: [svelte(), tailwind()],
  server: { port: 4321 },
  vite: {
    define: {
      "import.meta.env.PUBLIC_API_URL":
        JSON.stringify(process.env.PUBLIC_API_URL || "http://localhost:8000"),
    },
  },
});