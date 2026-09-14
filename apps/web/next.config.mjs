/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  // Served from https://deepsafehq.github.io/deepsafe-bench/
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || '',
};

export default nextConfig;
