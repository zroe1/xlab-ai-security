"use client";

import React from "react";
import Link from "next/link";
import styles from "./NextPageButton.module.css";
import { navigationItems, type DirectLink, type SubItem } from "../data/navigation";

interface NextPageButtonProps {
  currentSection: string;
  currentSlug: string;
}

// Helper function to flatten all navigation links into a sequential array
function getAllLinks(): DirectLink[] {
  const links: DirectLink[] = [];

  function extractLinks(items: SubItem[]): void {
    for (const item of items) {
      if (item.type === "link") {
        links.push(item);
      } else if (item.type === "folder") {
        extractLinks(item.items);
      }
    }
  }

  for (const section of navigationItems) {
    extractLinks(section.items);
  }

  return links;
}

const NextPageButton: React.FC<NextPageButtonProps> = ({ currentSection, currentSlug }) => {
  const allLinks = getAllLinks();
  const currentPath = `/${currentSection}/${currentSlug}`;

  // Find current page index
  const currentIndex = allLinks.findIndex((link) => link.href === currentPath);

  // Get next page
  const nextPage =
    currentIndex >= 0 && currentIndex < allLinks.length - 1 ? allLinks[currentIndex + 1] : null;

  // Don't render if there's no next page
  if (!nextPage) {
    return null;
  }

  return (
    <div className={styles.nextPageContainer}>
      <Link href={nextPage.href} className={styles.nextPageButton}>
        <div className={styles.nextPageContent}>
          <span className={styles.nextPageLabel}>Next</span>
          <span className={styles.nextPageTitle}>{nextPage.title}</span>
        </div>
        <div className={styles.nextPageIcon}>
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            xmlns="http://www.w3.org/2000/svg">
            <path
              d="M6 12L10 8L6 4"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
      </Link>
    </div>
  );
};

export default NextPageButton;
