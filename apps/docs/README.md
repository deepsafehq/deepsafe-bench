# DeepSafe API Documentation

Public API documentation for [DeepSafe](https://deepsafehq.github.io/deepsafe-bench), deployed to [deepsafehq.github.io/deepsafe-bench/docs](https://deepsafehq.github.io/deepsafe-bench/docs).

Built with [Fumadocs](https://fumadocs.vercel.app/) (Next.js + MDX).

## Getting Started

```bash
pnpm install
pnpm dev
```

Docs available at [http://localhost:3000](http://localhost:3000).

## Project Structure

```
├── app/                  # Next.js app directory
├── content/
│   └── docs/            # Documentation pages (MDX)
├── lib/
│   └── theme-config.ts  # Site configuration
└── public/              # Static assets
```

## Writing Documentation

Add MDX files to `content/docs/` to create new pages. The sidebar navigation is automatically generated based on file structure.
