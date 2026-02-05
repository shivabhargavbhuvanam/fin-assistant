/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // Design System Colors - Warm Neutral Palette
      colors: {
        // Primary palette (warm, editorial style)
        bg: '#F7F5F2',           // App background - warm off-white
        surface: '#FFFFFF',       // Cards, panels - pure white
        border: '#E6E2DC',        // Dividers, outlines - soft beige-gray
        textPrimary: '#111111',   // Main text - ink black
        textSecondary: '#5F5B57', // Metadata, reasoning - warm charcoal
        muted: '#8C877F',         // Hints, placeholders - stone gray
        accent: '#1C1C1C',        // Primary emphasis - near-black
      },
      // Font family
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
      },
      // Border radius
      borderRadius: {
        'xl': '0.75rem',
      },
      // Max width for layout
      maxWidth: {
        '6xl': '1200px',
      },
    },
  },
  plugins: [],
}
