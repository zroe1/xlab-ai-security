"use client";
import MainLayout from "@/components/MainLayout";
import Link from "next/link";
import React, { useState } from "react";

export default function Home() {
  // const [showTOC, setShowTOC] = useState(true);
  const tocItems = [
    { id: "1.1.", text: "1.1. Installation" },
    { id: "here", text: "1.1.1. Update existing AI security environment" },
  ];
  return (
    <MainLayout tocItems={tocItems}>
      {/* <h1 className="page-title">UChicago XLab AI Security Guide</h1>

      <div className="content-block">
        <p>
          Welcome to the AI Security Guide. This comprehensive resource is designed to help you
          understand and implement security best practices for artificial intelligence systems.
        </p>
      </div>

      <div className="content-block">
        <h2>Getting Started</h2>
        <p>
          Begin your journey with the{" "}
          <Link href="/getting-started/installation" className="text-link">
            Installation guide
          </Link>{" "}
          to set up your AI security toolkit.
        </p>
      </div>

      <div className="content-block">
        <h2>Featured Content</h2>
        <ul className="feature-list">
          <li>
            <Link href="/model-inference-attacks/stealing-model-weights" className="feature-link">
              Model Extraction Attacks
            </Link>
            <p>Learn how attackers can steal information about AI models</p>
          </li>
          <li>
            <Link href="/adversarial-examples/creating" className="feature-link">
              Creating Adversarial Inputs
            </Link>
            <p>Understand how to generate inputs that fool AI systems</p>
          </li>
        </ul>
      </div> */}

      <div className="content-container">
        <div className="content-wrapper">
          <h1 className="page-title">1.1. Installation</h1>

          <div className="content-block">
            <p>
              Setting up a secure AI development environment is the first step in building secure AI
              systems. This guide will walk you through the installation process for the UChicago
              XLab AI Security toolkit.
            </p>
          </div>

          <div className="content-block">
            <p>
              The <span className="inline-code">aisecsdk</span> installer and version management
              tool is the best way to download, install, and maintain your AI security development
              environment. Using the <span className="inline-code">aisecsdk</span> command after
              installation will help you check for updates and update your environment when
              necessary.
            </p>
          </div>

          <div className="content-block">
            <p>
              Depending on your operating system, you can install{" "}
              <span className="inline-code">aisecsdk</span> by following the instructions below:
            </p>
          </div>

          {/* Tabs for different OS instructions */}
          <div className="content-block">
            <div className="tabs">
              <button className="tab active">Linux or macOS</button>
              <button className="tab">Windows</button>
            </div>

            <div>
              <p>Enter the following command in terminal:</p>
              <div className="code-block">
                curl --proto 'https' --tlsv1.2 https://ai-sec.toolkit.org/install.sh -sSf | sh
              </div>
            </div>
          </div>

          <h2 className="section-title">1.1.1. Update existing AI security environment</h2>

          <div className="content-block">
            <p>You can run:</p>

            <div className="code-block">aisecsdk --version</div>

            <p>
              to both check if you already have the toolkit installed, and if so, which version. If
              you don't have it installed, go above and follow the instructions to install it. If
              you already have an AI security environment installed, then you can update the version
              by doing:
            </p>

            <div className="code-block">aisecsdk update</div>
          </div>

          {/* {children} */}
        </div>
      </div>
      {/* </div> */}

      {/* {showTOC && tocItems.length > 0 && (
        <div className="toc-sidebar">
          <div className="toc-header">
            <h3 className="toc-title">Contents</h3>
            <button onClick={() => setShowTOC(false)} className="toc-close">
              <svg
                width="20"
                height="20"
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
          </div>

          <nav className="toc-nav">
            <ul>
              {tocItems.map((item, index) => (
                <li key={index}>
                  <a href={`#${item.id}`} className="toc-link">
                    {item.text}
                  </a>
                </li>
              ))}
            </ul>
          </nav>
        </div>
      )} */}
    </MainLayout>
  );
}
