"use client";

import React, { useState, useEffect, useMemo } from "react";
import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { useTheme } from "@/contexts/ThemeContext";
import { searchIndex, type SearchIndexEntry } from "../data/searchIndex";

import {
  navigationItems,
  type Folder,
  type SubItem,
  type NavigationItem,
} from "../data/navigation";

// Search result interface
interface SearchResult {
  id: string;
  title: string;
  href: string;
  snippet: string;
  sectionTitle: string;
  sectionId: string;
}

// Search function using the generated index
const searchContent = (query: string): SearchResult[] => {
  if (!query.trim()) return [];

  const results: SearchResult[] = [];
  const lowercaseQuery = query.toLowerCase();

  for (const [href, content] of Object.entries(searchIndex)) {
    const typedContent = content as SearchIndexEntry;
    const titleMatch = typedContent.title.toLowerCase().includes(lowercaseQuery);
    const contentMatch = typedContent.content.toLowerCase().includes(lowercaseQuery);

    if (titleMatch || contentMatch) {
      // Find the snippet around the match
      let snippet = typedContent.content;
      if (contentMatch) {
        const matchIndex = typedContent.content.toLowerCase().indexOf(lowercaseQuery);
        const start = Math.max(0, matchIndex - 50);
        const end = Math.min(typedContent.content.length, matchIndex + query.length + 50);
        snippet = typedContent.content.slice(start, end);
        if (start > 0) snippet = "..." + snippet;
        if (end < typedContent.content.length) snippet = snippet + "...";
      } else {
        snippet = typedContent.content.slice(0, 100) + "...";
      }

      results.push({
        id: href,
        title: typedContent.title,
        href,
        snippet,
        sectionTitle: typedContent.sectionTitle,
        sectionId: typedContent.sectionId,
      });
    }
  }

  return results;
};

// Highlight search term in text
const highlightText = (text: string, query: string) => {
  if (!query.trim()) return text;

  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
  const parts = text.split(regex);

  return parts.map((part, index) =>
    regex.test(part) ? (
      <mark key={index} className="search-highlight">
        {part}
      </mark>
    ) : (
      part
    )
  );
};

// Search Results Component
const SearchResults = ({
  results,
  query,
  onResultClick,
}: {
  results: SearchResult[];
  query: string;
  onResultClick: () => void;
}) => {
  if (results.length === 0) {
    return (
      <div className="search-no-results">
        <div className="search-no-results-icon">
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2">
            <circle cx="11" cy="11" r="8"></circle>
            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
          </svg>
        </div>
        <p className="search-no-results-text">No results found for &ldquo;{query}&rdquo;</p>
        <p className="search-no-results-suggestion">Try different keywords or check spelling</p>
      </div>
    );
  }

  return (
    <div className="search-results">
      <div className="search-results-header">
        <span className="search-results-count">
          {results.length} result{results.length !== 1 ? "s" : ""} for &ldquo;{query}&rdquo;
        </span>
      </div>

      <div className="search-results-list">
        {results.map((result) => (
          <Link
            key={result.id}
            href={result.href}
            className="search-result-item"
            onClick={onResultClick}>
            <div className="search-result-header">
              <div className="search-result-title">{highlightText(result.title, query)}</div>
              <div className="search-result-section">{result.sectionTitle}</div>
            </div>
            <div className="search-result-snippet">{highlightText(result.snippet, query)}</div>
          </Link>
        ))}
      </div>
    </div>
  );
};

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

  // Memoize search results to avoid recalculating on every render
  const searchResults = useMemo(() => searchContent(searchText), [searchText]);
  const isSearching = searchText.trim().length > 0;

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

  const clearSearch = () => {
    setSearchText("");
  };

  const handleSearchResultClick = () => {
    setSearchText("");
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
            placeholder="Search content..."
            className="search-input"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
          />
          <div className="search-icon">
            {isSearching ? (
              <button
                onClick={clearSearch}
                className="search-clear-button"
                aria-label="Clear search">
                <svg
                  width="16"
                  height="16"
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
                <circle cx="11" cy="11" r="8"></circle>
                <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              </svg>
            )}
          </div>
        </div>
      </div>

      <div className="sidebar-nav">
        {isSearching ? (
          <SearchResults
            results={searchResults}
            query={searchText}
            onResultClick={handleSearchResultClick}
          />
        ) : (
          navigationItems.map((item) => (
            <NavItem
              key={item.id}
              item={item}
              activeItem={activeItem}
              setActiveItem={handleSetActiveItem}
            />
          ))
        )}
      </div>
    </div>
  );
};

export default Sidebar;
