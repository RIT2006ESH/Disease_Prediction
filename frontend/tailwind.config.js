/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#F3F6F4",
        ink: "#12211C",
        clinical: "#0B6E5C",
        alert: "#C13B2E",
        signal: "#2E8B57",
        line: "#D7DEDA",
      },
      fontFamily: {
        mono: ["JetBrains Mono", "monospace"],
        sans: ["Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
}