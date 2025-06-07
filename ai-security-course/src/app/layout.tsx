import "katex/dist/katex.min.css";
import "../styles/main.css";
import { ReactNode } from "react";

export const metadata = {
  title: "UChicago XLab AI Security Guide",
  description: "A comprehensive guide to AI security concepts and practices",
};

interface RootLayoutProps {
  children: ReactNode;
}

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
