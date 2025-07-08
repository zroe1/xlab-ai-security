"use client";

import React, { useState } from "react";
import styles from "./Dropdown.module.css";

interface DropdownProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const Dropdown: React.FC<DropdownProps> = ({ title, children, defaultOpen = false }) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  const toggleDropdown = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className={styles.dropdown}>
      <button
        className={styles.dropdownHeader}
        onClick={toggleDropdown}
        aria-expanded={isOpen}
        type="button"
      >
        <span className={styles.dropdownTitle}>{title}</span>
        <span className={`${styles.dropdownChevron} ${isOpen ? styles.expanded : ''}`}>
          <svg
            width="16"
            height="16"
            viewBox="0 0 16 16"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              d="M4 6L8 10L12 6"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </button>
      {isOpen && (
        <div className={styles.dropdownContent}>
          {children}
        </div>
      )}
    </div>
  );
};

export default Dropdown; 