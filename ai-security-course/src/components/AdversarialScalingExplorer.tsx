"use client";

import React, { useState, useEffect, useRef } from "react";
import styles from "./AdversarialScalingExplorer.module.css";

const AdversarialScalingExplorer: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [flopsPower, setFlopsPower] = useState(21); // 10^21 as default
  const [fid, setFid] = useState(1.65); // DG-20 quality as default
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Detect theme changes
  useEffect(() => {
    const updateTheme = () => {
      const isDark = document.body.getAttribute("data-theme") === "dark";
      setIsDarkMode(isDark);
    };

    // Initial check
    updateTheme();

    // Listen for theme changes
    const observer = new MutationObserver(updateTheme);
    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });

    return () => observer.disconnect();
  }, []);

  // Scaling law function ported from Python
  const predictAdversarialAccuracy = (flops: number, fid: number): number => {
    const A = 6.69;
    const B = 9.89;
    const E = 0.48;
    const alpha = 0.24;
    const beta = 0.23;
    const epsilon = 0.16;
    const zeta = -0.28;

    const slope = -0.7496;
    const intercept = 1.2575;

    // Calculate data quality-adjusted parameters
    const B_prime = Math.exp(Math.log(B) + Math.log(1 + fid) * zeta);
    const E_prime = Math.exp(Math.log(E) + Math.log(1 + fid) * epsilon);

    // Calculate optimal allocation parameters
    const G = Math.pow((alpha * A) / (beta * B_prime), 1 / (beta + alpha));
    const a = beta / (beta + alpha);
    const b = alpha / (beta + alpha);

    // Calculate optimal model size N* and dataset size D*
    const N_star = G * Math.pow(flops / 7822, a);
    const D_star = (1 / G) * Math.pow(flops / 7822, b);

    // Calculate predicted optimal loss
    const l_hat = A / Math.pow(N_star, alpha) + B_prime / Math.pow(D_star, beta) + E_prime;

    // Convert loss to accuracy
    const acc_pred = slope * l_hat + intercept;

    return Math.max(0, Math.min(1, acc_pred)); // Clamp between 0 and 1
  };

  // Draw the chart
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Theme-appropriate colors
    const backgroundColor = isDarkMode ? "#1a1a2e" : "#ffffff";
    const gridColor = isDarkMode ? "#333366" : "#e0e0e0";
    const textColor = isDarkMode ? "#ffffff" : "#333333";
    const axisColor = isDarkMode ? "#ffffff" : "#333333";

    // Clear canvas with theme-appropriate background
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Chart dimensions
    const margin = 60;
    const chartWidth = width - 2 * margin;
    const chartHeight = height - 2 * margin - 30; // Reduced bottom space

    // Coordinate transformations
    const minFlopsPower = 17;
    const maxFlopsPower = 30;
    const minAccuracy = 0.5;
    const maxAccuracy = 1.0;

    const xScale = (flopsPower: number) =>
      margin + ((flopsPower - minFlopsPower) / (maxFlopsPower - minFlopsPower)) * chartWidth;
    const yScale = (accuracy: number) =>
      margin + (1 - (accuracy - minAccuracy) / (maxAccuracy - minAccuracy)) * chartHeight;

    // Draw grid lines
    ctx.strokeStyle = gridColor;
    ctx.lineWidth = 0.5;

    // Vertical grid lines (FLOPs powers)
    for (let power = minFlopsPower; power <= maxFlopsPower; power += 2) {
      const x = xScale(power);
      ctx.beginPath();
      ctx.moveTo(x, margin);
      ctx.lineTo(x, margin + chartHeight);
      ctx.stroke();
    }

    // Horizontal grid lines (accuracy)
    for (let acc = 0.5; acc <= 1.0; acc += 0.1) {
      const y = yScale(acc);
      ctx.beginPath();
      ctx.moveTo(margin, y);
      ctx.lineTo(margin + chartWidth, y);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(margin, margin);
    ctx.lineTo(margin, margin + chartHeight);
    ctx.lineTo(margin + chartWidth, margin + chartHeight);
    ctx.stroke();

    // Draw the scaling law curve
    ctx.strokeStyle = "#8b1724";
    ctx.lineWidth = 3;
    ctx.beginPath();

    let firstPoint = true;
    for (let power = minFlopsPower; power <= maxFlopsPower; power += 0.1) {
      const flops = Math.pow(10, power);
      const accuracy = predictAdversarialAccuracy(flops, fid);
      const x = xScale(power);
      const y = yScale(accuracy);

      if (firstPoint) {
        ctx.moveTo(x, y);
        firstPoint = false;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Draw current position dot
    const currentFlops = Math.pow(10, flopsPower);
    const currentAccuracy = predictAdversarialAccuracy(currentFlops, fid);
    const currentX = xScale(flopsPower);
    const currentY = yScale(currentAccuracy);

    ctx.fillStyle = "#703d42";
    ctx.beginPath();
    ctx.arc(currentX, currentY, 8, 0, 2 * Math.PI);
    ctx.fill();

    // Add border to dot
    ctx.strokeStyle = backgroundColor === "#ffffff" ? "#333333" : "#ffffff";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw labels
    ctx.fillStyle = textColor;
    ctx.font = "12px Montserrat, sans-serif";

    // Y-axis labels
    for (let acc = 0.5; acc <= 1.0; acc += 0.1) {
      const y = yScale(acc);
      ctx.textAlign = "right";
      ctx.fillText(`${(acc * 100).toFixed(0)}%`, margin - 10, y + 4);
    }

    // X-axis labels
    for (let power = minFlopsPower; power <= maxFlopsPower; power += 2) {
      const x = xScale(power);
      ctx.textAlign = "center";
      ctx.fillText(`10^${power}`, x, margin + chartHeight + 12);
    }

    // Axis titles
    ctx.font = "14px Montserrat, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Training FLOPs", width / 2, height - 8);

    ctx.save();
    ctx.translate(20, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Adversarial Accuracy", 0, 0);
    ctx.restore();

    // Title
    ctx.font = "bold 20px Montserrat, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Adversarial Robustness Scaling Law", width / 2, 30);
  }, [flopsPower, fid, isDarkMode]);

  const currentFlops = Math.pow(10, flopsPower);
  const currentAccuracy = predictAdversarialAccuracy(currentFlops, fid);

  return (
    <div className={styles.container}>
      <canvas ref={canvasRef} width={800} height={600} className={styles.canvas} />

      {/* Controls */}
      <div className={styles.controls}>
        {/* FLOPs Slider */}
        <div className={styles.controlGroup}>
          <label className={styles.label}>
            Training FLOPs: 10^{flopsPower} ({currentFlops.toExponential(2)})
          </label>
          <input
            type="range"
            min={17}
            max={30}
            step={0.1}
            value={flopsPower}
            onChange={(e) => setFlopsPower(parseFloat(e.target.value))}
            className={styles.slider}
          />
          <div className={styles.sliderLabels}>
            <span>10^17</span>
            <span>10^30</span>
          </div>
        </div>

        {/* FID Slider */}
        <div className={styles.controlGroup}>
          <label className={styles.label}>
            Data Quality (FID): {fid.toFixed(2)}{" "}
            {fid < 2 ? "(Excellent)" : fid < 10 ? "(Good)" : "(Poor)"}
          </label>
          <input
            type="range"
            min={0.1}
            max={50}
            step={0.1}
            value={fid}
            onChange={(e) => setFid(parseFloat(e.target.value))}
            className={styles.slider}
          />
          <div className={styles.sliderLabels}>
            <span>0.1 (Perfect)</span>
            <span>50.0 (Poor)</span>
          </div>
        </div>

        {/* Results Display */}
        <div className={styles.results}>
          <h3 className={styles.resultsTitle}>Predicted Performance</h3>
          <div className={styles.metricsGrid}>
            <div className={styles.metric}>
              <div className={styles.metricValue}>{(currentAccuracy * 100).toFixed(1)}%</div>
              <div className={styles.metricLabel}>Adversarial Accuracy</div>
            </div>
            <div className={styles.metric}>
              <div className={styles.metricValue}>10^{flopsPower}</div>
              <div className={styles.metricLabel}>Training FLOPs</div>
            </div>
            <div className={styles.metric}>
              <div className={styles.metricValue}>{fid.toFixed(2)}</div>
              <div className={styles.metricLabel}>Data FID</div>
            </div>
          </div>
        </div>

        {/* Information */}
        <div className={styles.info}>
          <p>
            <strong>About:</strong> This interactive visualization shows the scaling law from
            &ldquo;Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies&rdquo;
            (Bartoldson et al., 2024). The curve shows how adversarial accuracy scales with compute
            (FLOPs) and data quality (FID). Lower FID = higher quality data.
          </p>
        </div>
      </div>
    </div>
  );
};

export default AdversarialScalingExplorer;
