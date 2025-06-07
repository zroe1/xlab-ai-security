import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      "@/components": path.resolve("src/components"),
      "@/lib": path.resolve("src/lib"),
      "@/styles": path.resolve("src/styles"),
      "@/contexts": path.resolve("src/contexts"),
    };
    return config;
  },
};

export default nextConfig;
