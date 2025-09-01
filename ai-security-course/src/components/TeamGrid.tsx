"use client";

import React from "react";
import styles from "./TeamGrid.module.css";

interface TeamGridProps {
  children: React.ReactNode;
}

const TeamGrid: React.FC<TeamGridProps> = ({ children }) => {
  return <div className={styles.grid}>{children}</div>;
};

export default TeamGrid;
