"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { useTheme } from "@/contexts/ThemeContext";

// Define types for navigation items
interface DirectLink {
  id: string;
  title: string;
  href: string;
  type: "link";
}

interface Folder {
  id: string;
  title: string;
  type: "folder";
  items: DirectLink[];
}

type SubItem = DirectLink | Folder;

interface NavigationItem {
  id: string;
  title: string;
  items: SubItem[];
}

// Sample navigation structure with nested folders
const navigationItems: NavigationItem[] = [
  {
    id: "1",
    title: "Getting Started",
    items: [
      { id: "1.1", title: "Installation", href: "/", type: "link" },
      { id: "1.2", title: "Hello World", href: "/getting-started/hello-world", type: "link" },
      { id: "1.3", title: "Debugging Programs", href: "/getting-started/debugging", type: "link" },
    ],
  },
  {
    id: "2",
    title: "Core Concepts",
    items: [
      { id: "2.1", title: "Threat Models", href: "/core-concepts/threat-models", type: "link" },
      {
        id: "2.2",
        title: "Attack Fundamentals",
        type: "folder",
        items: [
          {
            id: "2.2.1",
            title: "Attack Vectors",
            href: "/core-concepts/attack-vectors",
            type: "link",
          },
          {
            id: "2.2.2",
            title: "Attack Surfaces",
            href: "/core-concepts/attack-surfaces",
            type: "link",
          },
          {
            id: "2.2.3",
            title: "Risk Assessment",
            href: "/core-concepts/risk-assessment",
            type: "link",
          },
        ],
      },
      {
        id: "2.3",
        title: "Security Principles",
        type: "folder",
        items: [
          {
            id: "2.3.1",
            title: "Defense in Depth",
            href: "/core-concepts/defense-in-depth",
            type: "link",
          },
          {
            id: "2.3.2",
            title: "Least Privilege",
            href: "/core-concepts/least-privilege",
            type: "link",
          },
        ],
      },
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
        type: "link",
      },
      {
        id: "3.2",
        title: "Model Monitoring",
        href: "/defensive-techniques/model-monitoring",
        type: "link",
      },
    ],
  },
  {
    id: "4",
    title: "Adversarial Examples",
    items: [
      {
        id: "4.1",
        title: "Creating Adversarial Inputs",
        href: "/adversarial-examples/creating",
        type: "link",
      },
      {
        id: "4.2",
        title: "Defense Mechanisms",
        href: "/adversarial-examples/defense",
        type: "link",
      },
    ],
  },
];

// Helper function to normalize paths by removing trailing slashes
const normalizePath = (path: string): string => {
  return path === "/" ? "/" : path.replace(/\/$/, "");
};

// Helper function to check if a path matches any item recursively
const findActiveItemId = (items: SubItem[], normalizedPath: string): string | null => {
  for (const item of items) {
    if (item.type === "link") {
      if (normalizePath(item.href) === normalizedPath) {
        return item.id;
      }
    } else if (item.type === "folder") {
      const foundId = findActiveItemId(item.items, normalizedPath);
      if (foundId) return foundId;
    }
  }
  return null;
};

// Helper function to check if a folder contains the active path
const folderContainsActivePath = (folder: Folder, normalizedPath: string): boolean => {
  return folder.items.some((item) => normalizePath(item.href) === normalizedPath);
};

// Helper function to check if a section contains the active path
const sectionContainsActivePath = (section: NavigationItem, normalizedPath: string): boolean => {
  for (const item of section.items) {
    if (item.type === "link") {
      if (normalizePath(item.href) === normalizedPath) return true;
    } else if (item.type === "folder") {
      if (folderContainsActivePath(item, normalizedPath)) return true;
    }
  }
  return false;
};

const SubNavItem = ({
  item,
  activeItem,
  setActiveItem,
  normalizedPathname,
}: {
  item: SubItem;
  activeItem: string;
  setActiveItem: (id: string) => void;
  normalizedPathname: string;
}) => {
  // Always call hooks at the top level
  const [isExpanded, setIsExpanded] = useState(
    item.type === "folder" ? folderContainsActivePath(item, normalizedPathname) : false
  );

  useEffect(() => {
    if (item.type === "folder") {
      const shouldExpand = folderContainsActivePath(item, normalizedPathname);
      setIsExpanded(shouldExpand);
    }
  }, [normalizedPathname, item]);

  if (item.type === "link") {
    return (
      <Link
        key={item.id}
        href={item.href}
        className={`nav-item nav-link-custom ${activeItem === item.id ? "active" : ""}`}
        onClick={() => setActiveItem(item.id)}>
        <div className="nav-item-content">
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="nav-item-icon">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
          </svg>
          {item.id} {item.title}
        </div>
      </Link>
    );
  }

  // Folder type
  const toggleExpand = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="nav-subfolder">
      <div
        className={`nav-item nav-folder-toggle ${isExpanded ? "expanded" : ""}`}
        onClick={toggleExpand}>
        <div className="nav-item-content">
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="nav-item-icon">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2l5 0 2 3h9a2 2 0 0 1 2 2z"></path>
          </svg>
          {item.id} {item.title}
          <svg
            width="12"
            height="12"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`nav-folder-chevron ${isExpanded ? "expanded" : ""}`}>
            <polyline points="9 18 15 12 9 6"></polyline>
          </svg>
        </div>
      </div>
      {isExpanded && (
        <div className="nav-subfolder-items">
          {item.items.map((subItem) => (
            <Link
              key={subItem.id}
              href={subItem.href}
              className={`nav-item nav-subitem ${activeItem === subItem.id ? "active" : ""}`}
              onClick={() => setActiveItem(subItem.id)}>
              <div className="nav-item-content">
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="nav-item-icon">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                  <polyline points="14 2 14 8 20 8"></polyline>
                </svg>
                {subItem.id} {subItem.title}
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
};

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
  const normalizedPathname = normalizePath(pathname);
  const [isExpanded, setIsExpanded] = useState(sectionContainsActivePath(item, normalizedPathname));

  console.log(
    `NavItem ${item.id} - pathname: ${pathname}, normalizedPathname: ${normalizedPathname}, isExpanded: ${isExpanded}, activeItem: ${activeItem}`
  );

  // Update isExpanded when pathname changes
  useEffect(() => {
    const shouldExpand = sectionContainsActivePath(item, normalizedPathname);
    console.log(
      `NavItem ${item.id} useEffect - pathname: ${pathname}, normalizedPathname: ${normalizedPathname}, shouldExpand: ${shouldExpand}, current isExpanded: ${isExpanded}`
    );
    setIsExpanded(shouldExpand);
  }, [normalizedPathname, item]);

  const toggleExpand = () => {
    console.log(`NavItem ${item.id} - toggleExpand clicked, current isExpanded: ${isExpanded}`);
    setIsExpanded(!isExpanded);
  };

  const handleItemClick = (subItemId: string) => {
    console.log(`NavItem ${item.id} - item ${subItemId} clicked, setting activeItem`);
    setActiveItem(subItemId);
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
            <SubNavItem
              key={subItem.id}
              item={subItem}
              activeItem={activeItem}
              setActiveItem={handleItemClick}
              normalizedPathname={normalizedPathname}
            />
          ))}
        </div>
      )}
    </div>
  );
};

const Sidebar = () => {
  const { theme } = useTheme();
  const pathname = usePathname();
  const normalizedPathname = normalizePath(pathname);
  const [activeItem, setActiveItem] = useState("");
  const [searchText, setSearchText] = useState("");

  console.log(
    `Sidebar - pathname: ${pathname}, normalizedPathname: ${normalizedPathname}, activeItem: ${activeItem}`
  );

  useEffect(() => {
    console.log(
      `Sidebar useEffect - pathname changed to: ${pathname}, normalizedPathname: ${normalizedPathname}`
    );
    // Find the active item based on the current path
    for (const section of navigationItems) {
      const foundId = findActiveItemId(section.items, normalizedPathname);
      if (foundId) {
        console.log(
          `Sidebar useEffect - found matching item: ${foundId} for path: ${normalizedPathname}`
        );
        setActiveItem(foundId);
        return;
      }
    }
    console.log(`Sidebar useEffect - no matching item found for path: ${normalizedPathname}`);
  }, [normalizedPathname]);

  const handleSetActiveItem = (itemId: string) => {
    console.log(`Sidebar - setActiveItem called with: ${itemId}`);
    setActiveItem(itemId);
  };

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-logo-container">
          <Image
            src={theme === "dark" ? "/images/x_white.png" : "/images/x.png"}
            alt="UChicago XLab Logo"
            className="sidebar-logo"
            width={32}
            height={32}
          />
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
            setActiveItem={handleSetActiveItem}
          />
        ))}
      </div>
    </div>
  );
};

export default Sidebar;
