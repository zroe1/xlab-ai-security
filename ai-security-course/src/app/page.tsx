"use client";

import React from "react";
import Link from "next/link";
import { ThemeProvider, useTheme } from "@/contexts/ThemeContext";

const LandingPageContent = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <div className="landing-page-wrapper">
      {/* Simple header with just theme toggle and GitHub link */}
      <header className="landing-header">
        <div className="landing-header-content">
          <div className="landing-logo">
            <img src="/images/x.png" alt="UChicago XLab" className="landing-logo-img" />
            <span className="landing-logo-text">UChicago XLab</span>
          </div>
          <div className="landing-header-actions">
            <a
              href="https://github.com"
              className="github-link"
              target="_blank"
              rel="noopener noreferrer">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
              GitHub
            </a>
            <div className="theme-toggle-wrapper">
              <button
                className={`theme-toggle ${theme === "dark" ? "dark" : "light"}`}
                onClick={toggleTheme}
                aria-label="Toggle theme">
                <div className="theme-toggle-track">
                  <div className="theme-toggle-thumb">
                    <div className="theme-icon sun-icon">
                      <svg
                        width="12"
                        height="12"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2">
                        <circle cx="12" cy="12" r="5" />
                        <line x1="12" y1="1" x2="12" y2="3" />
                        <line x1="12" y1="21" x2="12" y2="23" />
                        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                        <line x1="1" y1="12" x2="3" y2="12" />
                        <line x1="21" y1="12" x2="23" y2="12" />
                        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                      </svg>
                    </div>
                    <div className="theme-icon moon-icon">
                      <svg
                        width="12"
                        height="12"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                      </svg>
                    </div>
                  </div>
                </div>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main landing page content */}
      <main className="landing-main">
        <div className="landing-page">
          {/* Hero Section */}
          <section className="hero-section">
            <h1 className="hero-title">There is no safety without security.</h1>
            <p className="hero-subtitle">
              The UChicago XLab AI Safety guide is built to prepare the next generation of AI
              engineers for the next generation of hackers.
            </p>
            <div className="hero-buttons">
              <Link href="/getting-started/installation" className="action-button primary">
                <img src="/images/x_white.png" alt="" className="button-icon xlab-icon" />
                View the Course
              </Link>
              <a href="https://github.com" className="action-button secondary">
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  className="button-icon github-icon">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
                </svg>
                Contribute on GitHub
              </a>
            </div>
          </section>

          {/* About Section */}
          <section className="about-section">
            <div className="section-content">
              <h2>About AI Security</h2>
              <p>
                While RLHF, constitutional AI, and other methods are often effective at improving
                the safety of AI products, they aren&apos;t robust to a variety of edge cases. A
                single clever prompt or adversarial suffix can undo months of safety training, while
                fine-tuning open weight models can easily bypass safeguards that appeared robust
                upon release.
              </p>
              <p>
                AI security research systematically exposes the gap between what AI models learn and
                what humans intend them to learn. Unlike other pillars of AI safety research which
                are pre-paradigm or theory-based, AI security research gives practical insights into
                how we can design safer models.
              </p>
            </div>
          </section>

          {/* Featured Research Section */}
          <section className="featured-research">
            <div className="section-content">
              <h2>Featured Research</h2>
              <div className="research-grid">
                <div className="research-tile">
                  <h3>Adversarial Examples in the Wild</h3>
                  <p>
                    Examining how adversarial perturbations affect real-world AI systems and the
                    implications for model robustness and security.
                  </p>
                  <div className="research-meta">
                    <span className="research-type">Image Classification</span>
                    <span className="research-year">2024</span>
                  </div>
                </div>

                <div className="research-tile">
                  <h3>Jailbreaking Large Language Models</h3>
                  <p>
                    A comprehensive analysis of prompt injection techniques and their effectiveness
                    against modern safety-aligned language models.
                  </p>
                  <div className="research-meta">
                    <span className="research-type">LLM Security</span>
                    <span className="research-year">2024</span>
                  </div>
                </div>

                <div className="research-tile">
                  <h3>Model Weight Extraction Attacks</h3>
                  <p>
                    Investigating techniques for extracting proprietary model parameters through
                    carefully crafted queries and inference-time attacks.
                  </p>
                  <div className="research-meta">
                    <span className="research-type">Model Privacy</span>
                    <span className="research-year">2023</span>
                  </div>
                </div>

                <div className="research-tile">
                  <h3>Robust Benchmarking for AI Safety</h3>
                  <p>
                    Developing evaluation frameworks that accurately measure model performance under
                    adversarial conditions and safety constraints.
                  </p>
                  <div className="research-meta">
                    <span className="research-type">Evaluation</span>
                    <span className="research-year">2023</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* About XLab Section */}
          <section className="xlab-section">
            <div className="section-content">
              <h2>About XLab</h2>
              <p>
                Founded in 2022 at the University of Chicago, the Existential Risk Laboratory (XLab)
                is an interdisciplinary research organization dedicated to the analysis and
                mitigation of risks that threaten human civilization&apos;s long-term survival.
              </p>
              <p>
                The legacy of existential risk work at the University of Chicago dates back to
                Enrico Fermi and the world&apos;s first nuclear chain reaction under the historic
                Stagg Field. XLab was founded in the same spirit of concern and commitment to
                mitigating the great threats of our time.
              </p>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default function Home() {
  return (
    <ThemeProvider>
      <LandingPageContent />
    </ThemeProvider>
  );
}
