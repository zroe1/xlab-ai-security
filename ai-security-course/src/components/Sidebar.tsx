"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";

// Define types for navigation items
interface SubItem {
  id: string;
  title: string;
  href: string;
}

interface NavigationItem {
  id: string;
  title: string;
  items: SubItem[];
}

// Sample navigation structure (will be replaced with dynamic data)
const navigationItems: NavigationItem[] = [
  {
    id: "1",
    title: "Getting Started",
    items: [
      { id: "1.1", title: "Installation", href: "/getting-started/installation" },
      { id: "1.2", title: "Hello World", href: "/getting-started/hello-world" },
      { id: "1.3", title: "Debugging Programs", href: "/getting-started/debugging" },
    ],
  },
  {
    id: "2",
    title: "Core Concepts",
    items: [
      { id: "2.1", title: "Threat Models", href: "/core-concepts/threat-models" },
      { id: "2.2", title: "Attack Vectors", href: "/core-concepts/attack-vectors" },
    ],
  },
  {
    id: "3",
    title: "Model Inference Attacks",
    items: [
      {
        id: "3.1",
        title: "Stealing Model Weights",
        href: "/model-inference-attacks/stealing-model-weights",
      },
      { id: "3.2", title: "Model Monitoring", href: "/defensive-techniques/model-monitoring" },
    ],
  },
  {
    id: "4",
    title: "Adversarial Examples",
    items: [
      { id: "4.1", title: "Creating Adversarial Inputs", href: "/adversarial-examples/creating" },
      { id: "4.2", title: "Defense Mechanisms", href: "/adversarial-examples/defense" },
    ],
  },
];

const NavItem = ({
  item,
  activeItem,
  setActiveItem,
}: {
  item: NavigationItem;
  activeItem: string;
  setActiveItem: (id: string) => void;
}) => {
  const pathname = usePathname();
  const [isExpanded, setIsExpanded] = useState(
    item.items.some((subItem) => subItem.href === pathname)
  );

  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="nav-section">
      <div className="nav-section-header" onClick={toggleExpand}>
        <button className="nav-section-toggle">
          {isExpanded ? (
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round">
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          ) : (
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round">
              <polyline points="9 18 15 12 9 6"></polyline>
            </svg>
          )}
        </button>
        <span className="nav-section-title">
          {item.id}. {item.title}
        </span>
      </div>

      {isExpanded && (
        <div className="nav-section-items">
          {item.items.map((subItem) => (
            <Link
              key={subItem.id}
              href={subItem.href}
              className={`nav-item nav-link-custom ${activeItem === subItem.id ? "active" : ""}`}
              onClick={() => setActiveItem(subItem.id)}>
              {subItem.id} {subItem.title}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
};

const Sidebar = () => {
  const pathname = usePathname();
  const [activeItem, setActiveItem] = useState("");
  const [searchText, setSearchText] = useState("");

  useEffect(() => {
    // Find the active item based on the current path
    for (const section of navigationItems) {
      for (const subItem of section.items) {
        if (subItem.href === pathname) {
          setActiveItem(subItem.id);
          return; // Exit once found
        }
      }
    }
  }, [pathname]);

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo-container">
          <img src="/images/x.png" alt="UChicago XLab Logo" className="sidebar-logo" />
          <div>
            <div className="sidebar-title">UChicago XLab</div>
            <div className="sidebar-subtitle">AI Security Guide</div>
          </div>
        </div>
      </div>

      <div className="search-container">
        <div className="search-input-wrapper">
          <input
            type="text"
            placeholder="Search"
            className="search-input"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
          />
          <div className="search-icon">
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
          </div>
        </div>
      </div>

      <div className="sidebar-nav">
        {navigationItems.map((item) => (
          <NavItem
            key={item.id}
            item={item}
            activeItem={activeItem}
            setActiveItem={setActiveItem}
          />
        ))}
      </div>
    </div>
  );
};

export default Sidebar;
