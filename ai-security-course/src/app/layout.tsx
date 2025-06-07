import "katex/dist/katex.min.css";
import "../styles/main.css";

export const metadata = {
  title: "UChicago XLab AI Security Guide",
  description: "A comprehensive guide to AI security concepts and practices",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
