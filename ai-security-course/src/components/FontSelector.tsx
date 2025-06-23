"use client";

import React, { useState, useEffect, useRef } from "react";

interface Font {
  name: string;
  family: string;
  googleFont?: boolean;
}

const fonts: Font[] = [
  { name: "Montserrat (Default)", family: "Montserrat, sans-serif" },
  { name: "Inter", family: "Inter, sans-serif", googleFont: true },
  { name: "Roboto", family: "Roboto, sans-serif" },
  { name: "Open Sans", family: "Open Sans, sans-serif", googleFont: true },
  { name: "Lato", family: "Lato, sans-serif", googleFont: true },
  { name: "Poppins", family: "Poppins, sans-serif", googleFont: true },
  { name: "Source Sans Pro", family: "Source Sans Pro, sans-serif", googleFont: true },
  { name: "Nunito Sans", family: "Nunito Sans, sans-serif" },
  { name: "Work Sans", family: "Work Sans, sans-serif", googleFont: true },
  { name: "Playfair Display", family: "Playfair Display, serif", googleFont: true },
  { name: "Merriweather", family: "Merriweather, serif", googleFont: true },
  { name: "Georgia", family: "Georgia, serif" },
  { name: "Times New Roman", family: "Times New Roman, serif" },
  { name: "JetBrains Mono", family: "JetBrains Mono, monospace", googleFont: true },
  { name: "Fira Code", family: "Fira Code, monospace", googleFont: true },
  { name: "Anonymous Pro", family: "Anonymous Pro, monospace" },
];

const FontSelector: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedFont, setSelectedFont] = useState(fonts[0]);
  const [loadedFonts, setLoadedFonts] = useState<Set<string>>(new Set());
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Load Google Fonts dynamically
  const loadGoogleFont = (fontFamily: string) => {
    if (loadedFonts.has(fontFamily)) return;

    const link = document.createElement("link");
    link.href = `https://fonts.googleapis.com/css2?family=${fontFamily.replace(
      / /g,
      "+"
    )}:wght@300;400;500;600;700&display=swap`;
    link.rel = "stylesheet";
    document.head.appendChild(link);

    setLoadedFonts((prev) => new Set(prev).add(fontFamily));
  };

  // Apply font to the entire document
  const applyFont = (font: Font) => {
    // Load Google Font if needed
    if (font.googleFont && font.family) {
      const fontName = font.family.split(",")[0].replace(/"/g, "");
      loadGoogleFont(fontName);
    }

    // Apply font to body and all elements
    document.body.style.fontFamily = font.family;

    // Also update CSS custom property if it exists
    document.documentElement.style.setProperty("--font-family", font.family);

    setSelectedFont(font);
    setIsOpen(false);
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Hide in production;
  if (process.env.NODE_ENV === "production") {
    return null;
  }

  return (
    <div className="font-selector-container" ref={dropdownRef}>
      {/* Dropdown Menu */}
      {isOpen && (
        <div className="font-selector-dropdown">
          <div className="font-selector-header">
            <h3>Font Selector</h3>
            <p>Development Tool</p>
          </div>
          <div className="font-selector-list">
            {fonts.map((font, index) => (
              <button
                key={index}
                className={`font-selector-item ${selectedFont.name === font.name ? "active" : ""}`}
                onClick={() => applyFont(font)}
                style={{ fontFamily: font.family }}>
                {font.name}
                {font.googleFont && <span className="font-badge">Google</span>}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Toggle Button */}
      <button
        className="font-selector-button"
        onClick={() => setIsOpen(!isOpen)}
        title="Font Selector (Dev Tool)">
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round">
          <polyline points="4,7 10,11 4,15"></polyline>
          <line x1="12" y1="19" x2="20" y2="19"></line>
          <line x1="12" y1="5" x2="20" y2="5"></line>
        </svg>
      </button>

      <style jsx>{`
        .font-selector-container {
          position: fixed;
          bottom: 20px;
          right: 20px;
          z-index: 9999;
        }

        .font-selector-button {
          width: 50px;
          height: 50px;
          border-radius: 50%;
          background: #8b1724;
          color: white;
          border: none;
          cursor: pointer;
          box-shadow: 0 4px 12px rgba(139, 23, 36, 0.3);
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.3s ease;
        }

        .font-selector-button:hover {
          background: #6d111c;
          transform: translateY(-2px);
          box-shadow: 0 6px 16px rgba(139, 23, 36, 0.4);
        }

        .font-selector-dropdown {
          position: absolute;
          bottom: 60px;
          right: 0;
          background: white;
          border-radius: 12px;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
          border: 1px solid #e0e0e0;
          width: 280px;
          max-height: 400px;
          overflow: hidden;
          animation: fadeInUp 0.2s ease-out;
        }

        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        .font-selector-header {
          padding: 16px;
          border-bottom: 1px solid #e0e0e0;
          background: #f8f9fa;
        }

        .font-selector-header h3 {
          margin: 0 0 4px 0;
          font-size: 16px;
          font-weight: 600;
          color: #333;
        }

        .font-selector-header p {
          margin: 0;
          font-size: 12px;
          color: #666;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .font-selector-list {
          max-height: 300px;
          overflow-y: auto;
          padding: 8px;
        }

        .font-selector-item {
          width: 100%;
          padding: 12px 16px;
          border: none;
          background: none;
          text-align: left;
          cursor: pointer;
          border-radius: 8px;
          margin-bottom: 2px;
          transition: all 0.2s ease;
          display: flex;
          align-items: center;
          justify-content: space-between;
          font-size: 14px;
        }

        .font-selector-item:hover {
          background: #f0f7ff;
          color: #8b1724;
        }

        .font-selector-item.active {
          background: #8b1724;
          color: white;
        }

        .font-badge {
          font-size: 10px;
          background: #4285f4;
          color: white;
          padding: 2px 6px;
          border-radius: 10px;
          text-transform: uppercase;
          letter-spacing: 0.3px;
        }

        .font-selector-item.active .font-badge {
          background: rgba(255, 255, 255, 0.3);
        }

        /* Dark theme support */
        body[data-theme="dark"] .font-selector-dropdown {
          background: #2a2a2a;
          border-color: #444;
        }

        body[data-theme="dark"] .font-selector-header {
          background: #1e1e1e;
          border-color: #444;
        }

        body[data-theme="dark"] .font-selector-header h3 {
          color: #e0e0e0;
        }

        body[data-theme="dark"] .font-selector-header p {
          color: #b0b0b0;
        }

        body[data-theme="dark"] .font-selector-item {
          color: #e0e0e0;
        }

        body[data-theme="dark"] .font-selector-item:hover {
          background: #3a3a3a;
          color: #8a535a;
        }
      `}</style>
    </div>
  );
};

export default FontSelector;
