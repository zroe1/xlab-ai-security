import React from "react";
import styles from "./FeedbackButton.module.css";

export interface FeedbackButtonProps {
  href: string;
  label?: string;
  newTab?: boolean;
}

const FeedbackButton: React.FC<FeedbackButtonProps> = ({
  href = "https://coda.io/form/XLab-AI-Security-Course-Feedback-Form_dR03g4WAlUD",
  label = "Give feedback on this section",
  newTab = true,
}) => {
  const target = newTab ? "_blank" : undefined;
  const rel = newTab ? "noopener noreferrer" : undefined;

  return (
    <div className={styles.feedbackContainer}>
      <a href={href} target={target} rel={rel} className={styles.feedbackButton}>
        <span className={styles.feedbackLabel}>{label}</span>
      </a>
    </div>
  );
};

export default FeedbackButton;
