"use client";

import React from "react";
import ThemeImage from "./ThemeImage";
import styles from "./OrganizationCard.module.css";

interface OrganizationCardProps {
  name: string;
  description: string;
  websiteUrl: string;
  lightLogoPath: string;
  darkLogoPath: string;
}

const OrganizationCard: React.FC<OrganizationCardProps> = ({
  name,
  description,
  websiteUrl,
  lightLogoPath,
  darkLogoPath,
}) => {
  return (
    <div className={styles.organizationCard}>
      <div className={styles.logoContainer}>
        <ThemeImage
          lightSrc={lightLogoPath}
          darkSrc={darkLogoPath}
          alt={`${name} logo`}
          className={styles.logo}
        />
      </div>
      <div className={styles.content}>
        <h3 className={styles.organizationName}>{name}</h3>
        <p className={styles.description}>{description}</p>
        <a
          href={websiteUrl}
          target="_blank"
          rel="noopener noreferrer"
          className={styles.websiteLink}>
          Visit Website →
        </a>
      </div>
    </div>
  );
};

export default OrganizationCard;
