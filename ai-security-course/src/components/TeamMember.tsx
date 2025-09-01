"use client";

import React from "react";
import styles from "./TeamMember.module.css";

interface TeamMemberProps {
  name: string;
  email: string;
  imageSrc: string;
}

const TeamMember: React.FC<TeamMemberProps> = ({ name, email, imageSrc }) => {
  return (
    <div className={styles.card}>
      <div className={styles.imageWrapper}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className={styles.image} src={imageSrc} alt={`${name} portrait`} />
      </div>
      <div className={styles.info}>
        <div className={styles.name}>{name}</div>
        <a className={styles.email} href={`mailto:${email}`}>
          {email}
        </a>
      </div>
    </div>
  );
};

export default TeamMember;
