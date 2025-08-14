/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0F172A',
        secondary: '#334155',
        muted: '#64748B',
        primary: {
          DEFAULT: '#4F46E5',
          hover: '#4338CA',
          focus: '#C7D2FE',
          subtle: '#EEF2FF',
        },
      },
      boxShadow: {
        card: '0 8px 24px rgba(15,23,42,0.06)',
      },
      borderRadius: {
        xl: '12px',
      }
    },
  },
  plugins: [],
}
