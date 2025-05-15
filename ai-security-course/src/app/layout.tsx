// File: src/app/layout.tsx
import { ReactNode } from "react";
import "../styles/main.css";

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
