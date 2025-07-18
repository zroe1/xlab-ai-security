// Types for navigation items
export interface DirectLink {
  id: string;
  title: string;
  href: string;
  type: "link";
}

export interface Folder {
  id: string;
  title: string;
  type: "folder";
  items: DirectLink[];
}

export type SubItem = DirectLink | Folder;

export interface NavigationItem {
  id: string;
  title: string;
  items: SubItem[];
}

// Navigation structure - single source of truth
export const navigationItems: NavigationItem[] = [
  {
    id: "1",
    title: "Getting Started",
    items: [
      { id: "1.1", title: "Welcome", href: "/getting-started/welcome", type: "link" },
      { id: "1.2", title: "Prerequisites", href: "/getting-started/prerequisites", type: "link" },
      { id: "1.3", title: "Setting Up Environment", href: "/getting-started/setup", type: "link" },
      {
        id: "1.4",
        title: "Running Coding Exercises",
        href: "/getting-started/running-coding-exercises",
        type: "link",
      },
    ],
  },
  {
    id: "2",
    title: "Adversarial Basics",
    items: [
      {
        id: "2.1",
        title: "Adversarial Basics Overview",
        href: "/adversarial/introduction",
        type: "link",
      },
      {
        id: "2.2",
        title: "Models and Data",
        href: "/adversarial/models-and-data",
        type: "link",
      },
      {
        id: "2.3",
        title: "White Box Attacks",
        type: "folder",
        items: [
          {
            id: "2.3.1",
            title: "FGSM and PGD",
            href: "/adversarial/adversarialimages",
            type: "link",
          },
          {
            id: "2.3.2",
            title: "Carlini & Wagner (C&W)",
            href: "/adversarial/cw",
            type: "link",
          },
        ],
      },
      {
        id: "2.4",
        title: "Black Box Attacks",
        type: "folder",
        items: [
          { id: "2.4.1", title: "Square Attack", href: "/adversarial/square-attack", type: "link" },
          {
            id: "2.4.2",
            title: "Ensemble Attacks",
            href: "/adversarial/ensemble-attacks",
            type: "link",
          },
        ],
      },
      {
        id: "2.5",
        title: "Defenses",
        type: "folder",
        items: [
          {
            id: "2.5.1",
            title: "Defensive Distillation",
            href: "/adversarial/defensive-distillation",
            type: "link",
          },
          {
            id: "2.5.2",
            title: "Adversarial Training",
            href: "/adversarial/adversarial-training",
            type: "link",
          },
        ],
      },
      {
        id: "2.6",
        title: "Benchmarks & State of the Art",
        type: "folder",
        items: [
          {
            id: "2.6.1",
            title: "RobustBench",
            href: "/adversarial/robustbench",
            type: "link",
          },
          {
            id: "2.6.2",
            title: "State of the Art",
            href: "/adversarial/sota",
            type: "link",
          },
          {
            id: "2.6.3",
            title: "Natural Limitations",
            href: "/adversarial/scaling-challenges",
            type: "link",
          },
        ],
      },
    ],
  },
  {
    id: "3",
    title: "LLM Jailbreaking",
    items: [
      {
        id: "3.1",
        title: "Introduction to Jailbreaks",
        href: "/jailbreaking/introduction",
        type: "link",
      },
      {
        id: "3.2",
        title: "Token-Level Attacks",
        type: "folder",
        items: [
          { id: "3.2.1", title: "GCG", href: "/jailbreaking/gcg", type: "link" },
          {
            id: "3.2.2",
            title: "AmpleGCG & Dense-to-Sparse Optimization",
            href: "/jailbreaking/amplegcg_adc",
            type: "link",
          },
        ],
      },
      {
        id: "3.3",
        title: "Prompt-Level Attacks",
        type: "folder",
        items: [
          { id: "3.3.1", title: "PAIR & TAP", href: "/jailbreaking/pair-tap", type: "link" },
          {
            id: "3.3.2",
            title: "GPTFuzzer & AutoDAN",
            href: "/jailbreaking/gptfuzzer",
            type: "link",
          },
        ],
      },
      {
        id: "3.4",
        title: "Agentic Attacks",
        type: "folder",
        items: [
          { id: "3.4.1", title: "AgentPoison", href: "/jailbreaking/agentpoison", type: "link" },
          {
            id: "3.4.2",
            title: "Commercial LLM Vulnerabilities",
            href: "/jailbreaking/commercial-vulnerabilities",
            type: "link",
          },
        ],
      },
      {
        id: "3.5",
        title: "Novel Attack Vectors",
        type: "folder",
        items: [
          {
            id: "3.5.1",
            title: "Visual Adversarial Jailbreaks",
            href: "/jailbreaking/visual",
            type: "link",
          },
          {
            id: "3.5.2",
            title: "Image Hijacks",
            href: "/jailbreaking/image-hijacks",
            type: "link",
          },
          {
            id: "3.5.3",
            title: "SolidGoldMagikarp",
            href: "/jailbreaking/solidgoldmagikarp",
            type: "link",
          },
          {
            id: "3.5.4",
            title: "Many-Shot Jailbreaking",
            href: "/jailbreaking/many-shot",
            type: "link",
          },
        ],
      },
    ],
  },
  {
    id: "4",
    title: "Model Tampering",
    items: [
      { id: "4.1", title: "Open-Weight Model Risks", href: "/tampering/overview", type: "link" },
      {
        id: "4.2",
        title: "Tampering Techniques",
        type: "folder",
        items: [
          {
            id: "4.2.1",
            title: "Refusal Direction Removal",
            href: "/tampering/refusal-direction",
            type: "link",
          },
          {
            id: "4.2.2",
            title: "Fine-tuning Attacks",
            href: "/tampering/fine-tuning",
            type: "link",
          },
          {
            id: "4.2.3",
            title: "Emergent Misalignment",
            href: "/tampering/emergent-misalignment",
            type: "link",
          },
        ],
      },
      {
        id: "4.3",
        title: "Tamper-Resistant Safeguards",
        href: "/tampering/safeguards",
        type: "link",
      },
      { id: "4.4", title: "Durability Evaluation", href: "/tampering/durability", type: "link" },
    ],
  },
  {
    id: "5",
    title: "Defenses & Guardrails",
    items: [
      { id: "5.1", title: "Defense Overview", href: "/defenses/overview", type: "link" },
      {
        id: "5.2",
        title: "Detection Methods",
        type: "folder",
        items: [
          { id: "5.2.1", title: "Perplexity Filters", href: "/defenses/perplexity", type: "link" },
          {
            id: "5.2.2",
            title: "Constitutional Classifiers",
            href: "/defenses/constitutional",
            type: "link",
          },
        ],
      },
      {
        id: "5.3",
        title: "Alignment Techniques",
        type: "folder",
        items: [
          { id: "5.3.1", title: "RLHF", href: "/defenses/rlhf", type: "link" },
          {
            id: "5.3.2",
            title: "CircuitBreakers",
            href: "/defenses/circuitbreakers",
            type: "link",
          },
          { id: "5.3.3", title: "SafeDecoding", href: "/defenses/safedecoding", type: "link" },
        ],
      },
      {
        id: "5.4",
        title: "Guardrail Systems",
        type: "folder",
        items: [
          { id: "5.4.1", title: "LlamaGuard", href: "/defenses/llamaguard", type: "link" },
          {
            id: "5.4.2",
            title: "Input/Output Filtering",
            href: "/defenses/filtering",
            type: "link",
          },
          { id: "5.4.3", title: "Safer APIs", href: "/defenses/safer-apis", type: "link" },
        ],
      },
      {
        id: "5.5",
        title: "Differential Privacy",
        href: "/defenses/differential-privacy",
        type: "link",
      },
    ],
  },
  {
    id: "6",
    title: "Information Extraction & Data Poisoning",
    items: [
      { id: "6.1", title: "Model Stealing Overview", href: "/extraction/overview", type: "link" },
      {
        id: "6.2",
        title: "Stealing Model Weights",
        href: "/extraction/stealing-weights",
        type: "link",
      },
      {
        id: "6.3",
        title: "Training Data Extraction",
        href: "/extraction/data-extraction",
        type: "link",
      },
      {
        id: "6.4",
        title: "Data Poisoning",
        href: "/extraction/data-poisoning",
        type: "link",
      },
      { id: "6.5", title: "Defenses", href: "/extraction/defenses", type: "link" },
    ],
  },
  {
    id: "7",
    title: "Resource and Future Directions",
    items: [
      {
        id: "7.1",
        title: "Jobs and Internships in AI Security",
        href: "/resources/jobs",
        type: "link",
      },
      {
        id: "7.2",
        title: "XLab Research",
        href: "/advanced/xlab-research",
        type: "link",
      },
      {
        id: "7.3",
        title: "Related Research Directions",
        href: "/advanced/related-research",
        type: "link",
      },
    ],
  },
];
