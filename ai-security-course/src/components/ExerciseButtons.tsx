import React from "react";
import styles from "./ExerciseButtons.module.css";

interface ExerciseButtonsProps {
  githubUrl?: string;
  colabUrl?: string;
}

const ExerciseButtons: React.FC<ExerciseButtonsProps> = ({ githubUrl, colabUrl }) => {
  if (!githubUrl && !colabUrl) {
    return null;
  }

  return (
    <div className={styles.exerciseButtons}>
      <div className={styles.exerciseButtonsHeader}>
        {/* <h3>Hands-on Exercises</h3>
        <p>Choose your preferred environment:</p> */}
      </div>
      <div className={styles.exerciseButtonsContainer}>
        {githubUrl && (
          <a
            href={githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className={`${styles.exerciseButton} ${styles.githubButton}`}>
            <div className={styles.exerciseButtonIcon}>
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="currentColor"
                xmlns="http://www.w3.org/2000/svg">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
            </div>
            <div className={styles.exerciseButtonContent}>
              <div className={styles.exerciseButtonTitle}>Download from GitHub</div>
              <div className={styles.exerciseButtonSubtitle}>Run locally on your machine</div>
            </div>
          </a>
        )}
        {colabUrl && (
          <a
            href={colabUrl}
            target="_blank"
            rel="noopener noreferrer"
            className={`${styles.exerciseButton} ${styles.colabButton}`}>
            <div className={styles.exerciseButtonIcon}>
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="currentColor"
                xmlns="http://www.w3.org/2000/svg">
                <path d="M16.9414 4.9757a7.033 7.033 0 0 0-4.9308 2.0324 7.033 7.033 0 0 0-.1232 9.8068l2.395-2.395a3.6455 3.6455 0 0 1 5.1497-5.1478l2.397-2.3989a7.033 7.033 0 0 0-4.8877-1.9375zm-9.8831 0a7.033 7.033 0 0 0-4.8878 1.9375l2.397 2.4a3.6455 3.6455 0 0 1 5.1497 5.1478l2.395 2.395a7.033 7.033 0 0 0-.1232-9.8068 7.033 7.033 0 0 0-4.9307-2.0324z" />
              </svg>
            </div>
            <div className={styles.exerciseButtonContent}>
              <div className={styles.exerciseButtonTitle}>Open in Google Colab</div>
              <div className={styles.exerciseButtonSubtitle}>Run in the cloud for free</div>
            </div>
          </a>
        )}
      </div>
    </div>
  );
};

export default ExerciseButtons;
