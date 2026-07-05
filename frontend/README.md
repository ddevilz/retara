# magenta-retain-frontend

React 19 + TypeScript + Vite frontend for Magenta Retain. Tailwind (magenta/ink theme),
react-router-dom, recharts, `@microsoft/fetch-event-source` for SSE streams from the backend.

## Scripts

- `npm run dev` — Vite dev server on `:5173`, proxies `/api` to `http://localhost:8000`.
- `npm run build` — type-check (`tsc -b`) then production build to `dist/`.
- `npm run typecheck` — `tsc --noEmit`.
- `npm run test` — Vitest (jsdom + Testing Library) once.
- `npm run preview` — preview the production build.

## Layout

- `src/App.tsx` — shell with nav (`Overview` / `Customers` / `Run-one` / `Negotiation`).
- `src/pages/` — route pages (placeholders until later tasks flesh them out).
- `src/test/setup.ts` — Vitest setup (`@testing-library/jest-dom` matchers).
