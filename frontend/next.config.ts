import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Proxy is optional — frontend calls the Python API directly via env var.
  // Add rewrites here if you prefer same-origin routing in production.
};

export default nextConfig;
