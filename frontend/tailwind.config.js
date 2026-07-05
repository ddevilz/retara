/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        magenta: {
          DEFAULT: "#E20074",
          600: "#C2005F",
          300: "#FF4FA8",
        },
        ink: {
          900: "#0B0B0F",
          800: "#14141B",
          700: "#1E1E28",
          600: "#2A2A38",
        },
      },
    },
  },
  plugins: [],
};
