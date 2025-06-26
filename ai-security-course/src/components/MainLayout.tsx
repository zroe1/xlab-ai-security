"use client";

import React, { useState, useRef, useEffect, ReactNode, useCallback } from "react";
import Sidebar from "./Sidebar";
import FontSelector from "./FontSelector";
import { ThemeProvider, useTheme } from "../contexts/ThemeContext";

interface TocItem {
  id: string;
  text: string;
}

interface LayoutProps {
  children: ReactNode;
  tocItems?: TocItem[];
}

const LayoutContent = ({ children, tocItems = [] }: LayoutProps) => {
  const { theme, toggleTheme } = useTheme();
  const [showTOC, setShowTOC] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(280); // Default width
  const [isResizing, setIsResizing] = useState(false);
  const [activeSection, setActiveSection] = useState<string>("");

  const appRef = useRef<HTMLDivElement>(null);
  const resizeHandleRef = useRef<HTMLDivElement>(null);

  // Track scroll position to highlight active TOC item
  useEffect(() => {
    if (tocItems.length === 0) return;

    const handleScroll = () => {
      const sections = tocItems.map((item) => document.getElementById(item.id)).filter(Boolean);
      const scrollPosition = window.scrollY + 100; // Offset for header

      for (let i = sections.length - 1; i >= 0; i--) {
        const section = sections[i];
        if (section && section.offsetTop <= scrollPosition) {
          setActiveSection(tocItems[i].id);
          break;
        }
      }
    };

    window.addEventListener("scroll", handleScroll);
    handleScroll(); // Set initial active section

    return () => window.removeEventListener("scroll", handleScroll);
  }, [tocItems]);

  // Toggle sidebar visibility
  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  // Handle resize functionality
  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing || !appRef.current) return;

      const appRect = appRef.current.getBoundingClientRect();
      const newWidth = Math.max(234, Math.min(500, e.clientX - appRect.left));

      // Update state
      setSidebarWidth(newWidth);

      // Update the position of the resize handle
      if (resizeHandleRef.current) {
        resizeHandleRef.current.style.left = `${newWidth}px`;
      }
    },
    [isResizing]
  );

  const stopResizing = useCallback(() => {
    setIsResizing(false);
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", stopResizing);

    // Remove the active class
    if (resizeHandleRef.current) {
      resizeHandleRef.current.classList.remove("active");
    }
  }, [handleMouseMove]);

  const startResizing = (e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault(); // Prevent default behavior
    setIsResizing(true);

    // Add event listeners to document instead of the handle itself
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", stopResizing);

    // Add a class to the resize handle to show it's active
    if (resizeHandleRef.current) {
      resizeHandleRef.current.classList.add("active");
    }
  };

  // Clean up event listeners when component unmounts
  useEffect(() => {
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", stopResizing);
    };
  }, [handleMouseMove, stopResizing]);

  // Reset resize handle position when sidebar is collapsed/expanded
  useEffect(() => {
    if (resizeHandleRef.current) {
      resizeHandleRef.current.style.left = sidebarCollapsed ? "0px" : `${sidebarWidth}px`;
    }
  }, [sidebarCollapsed, sidebarWidth]);

  // Add an effect to update handleMouseMove when isResizing changes
  useEffect(() => {
    // This is necessary because the handleMouseMove function closure
    // captures the initial value of isResizing
    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", stopResizing);
    } else {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", stopResizing);
    }
  }, [isResizing, handleMouseMove, stopResizing]);

  return (
    <div className="app-container" ref={appRef}>
      {/* Sidebar with dynamic width */}
      <div
        className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}
        style={{ width: sidebarCollapsed ? 0 : `${sidebarWidth}px` }}>
        <Sidebar />
      </div>

      {/* Resize Handle */}
      <div
        className="resize-handle"
        ref={resizeHandleRef}
        onMouseDown={startResizing}
        style={{ left: sidebarCollapsed ? "0px" : `${sidebarWidth}px` }}></div>

      {/* Main Content Area */}
      <div className="main-content">
        {/* Top Navigation Bar */}
        <header className="main-header">
          <button className="menu-button" onClick={toggleSidebar}>
            {sidebarCollapsed ? (
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round">
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
              </svg>
            ) : (
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            )}
          </button>

          <div className="header-actions">
            {sidebarCollapsed && (
              <button className="header-action" onClick={toggleSidebar} title="Open Navigation">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round">
                  <line x1="3" y1="12" x2="21" y2="12"></line>
                  <line x1="3" y1="6" x2="21" y2="6"></line>
                  <line x1="3" y1="18" x2="21" y2="18"></line>
                </svg>
              </button>
            )}
            <div className="theme-toggle-wrapper">
              <button
                className={`theme-toggle ${theme === "dark" ? "dark" : "light"}`}
                onClick={toggleTheme}
                title="Toggle Theme"
                aria-label="Toggle Theme">
                <div className="theme-toggle-track">
                  <div className="theme-toggle-thumb">
                    <div className="theme-icon sun-icon">
                      <svg
                        width="12"
                        height="12"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeLinejoin="round">
                        <circle cx="12" cy="12" r="5"></circle>
                        <line x1="12" y1="1" x2="12" y2="3"></line>
                        <line x1="12" y1="21" x2="12" y2="23"></line>
                        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                        <line x1="1" y1="12" x2="3" y2="12"></line>
                        <line x1="21" y1="12" x2="23" y2="12"></line>
                        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
                      </svg>
                    </div>
                    <div className="theme-icon moon-icon">
                      <svg
                        width="12"
                        height="12"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2.5"
                        strokeLinecap="round"
                        strokeLinejoin="round">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
                      </svg>
                    </div>
                  </div>
                </div>
              </button>
            </div>
            <a
              href="https://github.com/zroe1/xlab-ai-security"
              className="github-link"
              title="GitHub"
              target="_blank"
              rel="noopener noreferrer">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round">
                <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path>
              </svg>
            </a>
            {!showTOC && tocItems.length > 0 && (
              <button
                className="toc-button"
                onClick={() => setShowTOC(true)}
                title="Open Table of Contents">
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <line x1="9" y1="9" x2="15" y2="9"></line>
                  <line x1="9" y1="12" x2="15" y2="12"></line>
                  <line x1="9" y1="15" x2="15" y2="15"></line>
                </svg>
              </button>
            )}
          </div>
        </header>

        {/* Content Area with Scroll */}
        <div className="content-container">
          <div className="content-wrapper">{children}</div>
        </div>
      </div>

      {/* Table of Contents Sidebar */}
      {showTOC && tocItems.length > 0 && (
        <div className="toc-sidebar">
          <div className="toc-header">
            <h3 className="toc-title">Contents</h3>
            <button onClick={() => setShowTOC(false)} className="toc-close">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>

          <nav className="toc-nav">
            <ul>
              {tocItems.map((item, index) => (
                <li key={index}>
                  <a
                    href={`#${item.id}`}
                    className={`toc-link ${activeSection === item.id ? "active" : ""}`}>
                    {item.text}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      )}

      {/* Font Selector - Development Tool */}
      <FontSelector />
    </div>
  );
};

const MainLayout = ({ children, tocItems = [] }: LayoutProps) => {
  return (
    <ThemeProvider>
      <LayoutContent tocItems={tocItems}>{children}</LayoutContent>
    </ThemeProvider>
  );
};

export default MainLayout;
