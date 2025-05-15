"use client";

import React, { useState, useRef, useEffect } from "react";
import Sidebar from "./Sidebar";

const MainLayout = ({ children, tocItems }) => {
  const [showTOC, setShowTOC] = useState(true);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(280); // Default width
  const [isResizing, setIsResizing] = useState(false);

  const appRef = useRef(null);
  const resizeHandleRef = useRef(null);

  // Toggle sidebar visibility
  const toggleSidebar = () => {
    setSidebarCollapsed(!sidebarCollapsed);
  };

  // Handle resize functionality
  const startResizing = (e) => {
    e.preventDefault(); // Prevent default behavior
    setIsResizing(true);

    // Add event listeners to document instead of the handle itself
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", stopResizing);

    // Add a class to the resize handle to show it's active
    if (resizeHandleRef.current) {
      resizeHandleRef.current.classList.add("active");
    }

    // For debugging
    console.log("Start resizing");
  };

  const handleMouseMove = (e) => {
    if (!isResizing || !appRef.current) return;

    const appRect = appRef.current.getBoundingClientRect();
    const newWidth = Math.max(200, Math.min(500, e.clientX - appRect.left));

    // For debugging
    console.log("Resizing to width:", newWidth);

    // Update state
    setSidebarWidth(newWidth);

    // Update the position of the resize handle
    if (resizeHandleRef.current) {
      resizeHandleRef.current.style.left = `${newWidth}px`;
    }
  };

  const stopResizing = () => {
    setIsResizing(false);
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", stopResizing);

    // Remove the active class
    if (resizeHandleRef.current) {
      resizeHandleRef.current.classList.remove("active");
    }

    // For debugging
    console.log("Stop resizing");
  };

  // Clean up event listeners when component unmounts
  useEffect(() => {
    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", stopResizing);
    };
  }, []);

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
  }, [isResizing]);

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
          <button className="sidebar-toggle" onClick={toggleSidebar}>
            <span className="sr-only">{sidebarCollapsed ? "Open" : "Close"} Sidebar</span>
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
            <a href="#" className="header-action">
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
            <a href="#" className="header-action">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7 10 12 15 17 10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
            </a>
            <a href="#" className="header-action">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round">
                <path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path>
              </svg>
            </a>
            <a href="#" className="header-action">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
              </svg>
            </a>
          </div>
        </header>

        {/* Content Area with Scroll */}
        <div className="content-container">
          <div className="content-wrapper">
            <h1 className="page-title">1.1. Installation</h1>

            <div className="content-block">
              <p>
                Setting up a secure AI development environment is the first step in building secure
                AI systems. This guide will walk you through the installation process for the
                UChicago XLab AI Security toolkit.
              </p>
            </div>

            <div className="content-block">
              <p>
                The <span className="inline-code">aisecsdk</span> installer and version management
                tool is the best way to download, install, and maintain your AI security development
                environment. Using the <span className="inline-code">aisecsdk</span> command after
                installation will help you check for updates and update your environment when
                necessary.
              </p>
            </div>

            <div className="content-block">
              <p>
                Depending on your operating system, you can install{" "}
                <span className="inline-code">aisecsdk</span> by following the instructions below:
              </p>
            </div>

            {/* Tabs for different OS instructions */}
            <div className="content-block">
              <div className="tabs">
                <button className="tab active">Linux or macOS</button>
                <button className="tab">Windows</button>
              </div>

              <div>
                <p>Enter the following command in terminal:</p>
                <div className="code-block">
                  curl --proto 'https' --tlsv1.2 https://ai-sec.toolkit.org/install.sh -sSf | sh
                </div>
              </div>
            </div>

            <h2 className="section-title">1.1.1. Update existing AI security environment</h2>

            <div className="content-block">
              <p>You can run:</p>

              <div className="code-block">aisecsdk --version</div>

              <p>
                to both check if you already have the toolkit installed, and if so, which version.
                If you don't have it installed, go above and follow the instructions to install it.
                If you already have an AI security environment installed, then you can update the
                version by doing:
              </p>

              <div className="code-block">aisecsdk update</div>
            </div>

            {children}
          </div>
        </div>
      </div>

      {/* Table of Contents Sidebar */}
      {showTOC && (
        <div className="toc-sidebar">
          <div className="toc-header">
            <h3 className="toc-title">Contents</h3>
            <button onClick={() => setShowTOC(false)} className="toc-close">
              <span className="sr-only">Close TOC</span>
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
              <li>
                <a href="#" className="toc-link active">
                  1.1.1. Update existing AI security environment
                </a>
              </li>
              <li>
                <a href="#" className="toc-link">
                  1.1.2. AI Security Playground
                </a>
              </li>
            </ul>
          </nav>
        </div>
      )}
    </div>
  );
};

export default MainLayout;
