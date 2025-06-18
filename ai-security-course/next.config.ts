import type { NextConfig } from "next";
import path from "path";
import { generateSearchIndex } from "./scripts/generateSearchIndex";

const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: {
    unoptimized: true,
  },
  webpack: (config, { dev, isServer }) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      "@/components": path.resolve("src/components"),
      "@/lib": path.resolve("src/lib"),
      "@/styles": path.resolve("src/styles"),
      "@/contexts": path.resolve("src/contexts"),
    };
    // Generate search index during build
    if (!dev && isServer) {
      generateSearchIndex();
    }
    return config;
  },
};

export default nextConfig;
