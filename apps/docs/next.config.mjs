import { createMDX } from 'fumadocs-mdx/next'

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  // Served from https://deepsafehq.github.io/deepsafe-bench/docs/, so assets
  // must carry that prefix or every /_next/* request 404s.
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || '',
  images: { unoptimized: true },
  reactStrictMode: true,
}

const withMDX = createMDX()

export default withMDX(nextConfig)

