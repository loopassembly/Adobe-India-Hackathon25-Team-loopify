/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: '#4F46E5',
        primaryHover: '#4338CA',
        primaryFocus: '#C7D2FE',
        subtle: '#EEF2FF',
        ink: '#0F172A',
        secondary: '#334155',
        muted: '#64748B',
        card: '#F8FAFC',
        border: '#E2E8F0',
        success: '#059669',
        warning: '#D97706',
        error: '#DC2626',
        info: '#0EA5E9',
      },
      boxShadow: {
        card: '0 8px 24px rgba(15,23,42,0.06)',
      },
      borderRadius: {
        'xl2': '1rem',
      }
    },
  },
  plugins: [],
}