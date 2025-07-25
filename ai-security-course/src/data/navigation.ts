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
      { id: "1.3", title: "Setting Up Environment", href: "/getting-started/set-up", type: "link" },
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
            href: "/jailbreaking/gptfuzzer_autodan",
            type: "link",
          },
        ],
      },
      {
        id: "3.4",
        title: "Novel Attack Vectors",
        type: "folder",
        items: [
          {
            id: "3.4.1",
            title: "Visual Jailbreaks",
            href: "/jailbreaking/visual-jailbreaks",
            type: "link",
          },
          {
            id: "3.4.2",
            title: "Many-Shot Jailbreaking",
            href: "/jailbreaking/many-shot-jailbreaking",
            type: "link",
          },
          {
            id: "3.4.3",
            title: "Attacking Safeguard Pipelines",
            href: "/jailbreaking/stack",
            type: "link",
          },
          {
            id: "3.4.4",
            title: "Extras ",
            href: "/jailbreaking/extras",
            type: "link",
          },
        ],
      },
      {
        id: "3.5",
        title: "Defenses",
        type: "folder",
        items: [
          {
            id: "3.5.1",
            title: "Perplexity Filters & Baseline Defenses",
            href: "/jailbreaking/ppl-baseline",
            type: "link",
          },
          {
            id: "3.5.2",
            title: "Llama Guard",
            href: "/jailbreaking/llama-guard",
            type: "link",
          },
          {
            id: "3.5.3",
            title: "SafeDecoding",
            href: "/jailbreaking/safe-decoding",
            type: "link",
          },
          {
            id: "3.5.4",
            title: "SmoothLLM",
            href: "/jailbreaking/smooth-llm",
            type: "link",
          },
          {
            id: "3.5.5",
            title: "Constitutional Classifiers",
            href: "/jailbreaking/constitutional-classifiers",
            type: "link",
          },
          {
            id: "3.5.6",
            title: "Representation Engineering & Circuit Breakers",
            href: "/jailbreaking/circuit-breakers",
            type: "link",
          },
          {
            id: "3.5.7",
            title: "Deep Safety Alignment (pt. 1)",
            href: "/jailbreaking/deep-safety-alignment-1",
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
        title: "Tampering Defenses",
        type: "folder",
        items: [
          {
            id: "4.3.1",
            title: "Tamper-Resistant Safeguards",
            href: "/tampering/tamper-resistant-safeguards",
            type: "link",
          },
          {
            id: "4.3.2",
            title: "Unlearning & Distillation",
            href: "/tampering/unlearning-distillation",
            type: "link",
          },
        ],
      },
      { id: "4.4", title: "Durability Evaluation", href: "/tampering/durability", type: "link" },
      {
        id: "4.5",
        title: "Deep Safety Alignment (pt. 2)",
        href: "/tampering/deep-safety-alignment-2",
        type: "link",
      },
    ],
  },
  {
    id: "5",
    title: "Information Extraction & Data Poisoning",
    items: [
      { id: "5.1", title: "Model Stealing", href: "/extraction/model-stealing", type: "link" },
      {
        id: "5.2",
        title: "Training Data Extraction",
        href: "/extraction/data-extraction",
        type: "link",
      },
      {
        id: "5.3",
        title: "Data Poisoning & Backdooring",
        href: "/extraction/data-poisoning",
        type: "link",
      },
      { id: "5.4", title: "Defenses", href: "/extraction/defenses", type: "link" },
    ],
  },
  {
    id: "6",
    title: "Extra Topics",
    items: [
      { id: "6.1", title: "Agentic Attacks", href: "/defenses/agentic-attacks", type: "link" },
      { id: "6.2", title: "Jailbreak Tax", href: "/defenses/jailbreak-tax", type: "link" },
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
