"use client";

import React from "react";
import { useTheme } from "@/contexts/ThemeContext";

interface ThemeImageProps {
  lightSrc: string;
  darkSrc: string;
  alt: string;
  style?: React.CSSProperties;
  className?: string;
}

const ThemeImage: React.FC<ThemeImageProps> = ({ lightSrc, darkSrc, alt, style, className }) => {
  const { theme } = useTheme();
  const src = theme === "dark" ? darkSrc : lightSrc;

  return <img src={src} alt={alt} style={style} className={className} />;
};

export default ThemeImage;
