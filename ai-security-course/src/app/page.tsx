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
        </div>
      </div>
    </MainLayout>
  );
}
