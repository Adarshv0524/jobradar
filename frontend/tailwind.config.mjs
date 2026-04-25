/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,js,jsx,ts,tsx,svelte}"],
  theme: {
    extend: {
      fontFamily: {
        sans:  ['"Inter"', "system-ui", "sans-serif"],
        mono:  ['"JetBrains Mono"', "monospace"],
      },
      colors: {
        bg:       "#0d0e10",
        surface:  "#13151a",
        elevated: "#1a1d24",
        card:     "#1e2028",
        border:   "#2a2d36",
        "border-hi": "#3a3d48",
        accent:   "#f59e0b",
        green:    "#22c55e",
        blue:     "#60a5fa",
        red:      "#f87171",
        "text-hi": "#f1f2f4",
        text:     "#c9ccd4",
        "text-lo": "#8891a4",
        "text-dim": "#555b6e",
      },
      spacing: { "18": "4.5rem", "88": "22rem" },
      borderRadius: { DEFAULT: "6px" },
    },
  },
  plugins: [],
};