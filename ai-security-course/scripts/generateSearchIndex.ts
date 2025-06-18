import fs from "fs";
import path from "path";
import matter from "gray-matter";

interface SearchIndexEntry {
  title: string;
  content: string;
  sectionTitle: string;
  sectionId: string;
}

type SearchIndex = Record<string, SearchIndexEntry>;

// Function to strip markdown formatting and clean content
function cleanMarkdownContent(content: string): string {
  return (
    content
      // Remove code blocks
      .replace(/```[\s\S]*?```/g, "")
      // Remove inline code
      .replace(/`([^`]+)`/g, "$1")
      // Remove headers
      .replace(/^#{1,6}\s+/gm, "")
      // Remove links but keep text
      .replace(/\[([^\]]+)\]\([^\)]+\)/g, "$1")
      // Remove emphasis
      .replace(/[*_]{1,2}([^*_]+)[*_]{1,2}/g, "$1")
      // Remove extra whitespace
      .replace(/\s+/g, " ")
      .trim()
  );
}

// Function to map file paths to URLs and section info
function getUrlAndSection(
  filePath: string,
  contentDir: string
): { url: string; sectionTitle: string; sectionId: string } {
  const relativePath = path.relative(contentDir, filePath);
  const pathParts = relativePath.split(path.sep);

  // Handle root level files
  if (pathParts.length === 1) {
    const filename = path.basename(pathParts[0], ".mdx");
    if (filename === "installation") {
      return {
        url: "/",
        sectionTitle: "Getting Started",
        sectionId: "1",
      };
    }
    return {
      url: `/${filename}`,
      sectionTitle: "Getting Started",
      sectionId: "1",
    };
  }

  // Handle nested files
  const sectionFolder = pathParts[0];
  const filename = path.basename(pathParts[pathParts.length - 1], ".mdx");

  // Map section folders to navigation structure
  const sectionMap: Record<string, { title: string; id: string }> = {
    "getting-started": { title: "Getting Started", id: "1" },
    adversarial: { title: "Adversarial Examples", id: "2" },
    "adversarial-examples": { title: "Adversarial Examples", id: "2" },
    extraction: { title: "Model Extraction", id: "3" },
    "model-extraction": { title: "Model Extraction", id: "3" },
    "model-inference-attacks": { title: "Model Extraction", id: "3" },
    jailbreaking: { title: "LLM Jailbreaking", id: "4" },
    "llm-jailbreaking": { title: "LLM Jailbreaking", id: "4" },
    tampering: { title: "Model Tampering", id: "5" },
    "model-tampering": { title: "Model Tampering", id: "5" },
    defenses: { title: "Defenses & Guardrails", id: "6" },
    advanced: { title: "Advanced Topics", id: "7" },
  };

  const sectionInfo = sectionMap[sectionFolder] || { title: "Other", id: "0" };

  return {
    url: `/${sectionFolder}/${filename}`,
    sectionTitle: sectionInfo.title,
    sectionId: sectionInfo.id,
  };
}

// Function to recursively scan directory for MDX files
function findMdxFiles(dir: string): string[] {
  const files: string[] = [];

  const items = fs.readdirSync(dir);

  for (const item of items) {
    const fullPath = path.join(dir, item);
    const stat = fs.statSync(fullPath);

    if (stat.isDirectory()) {
      files.push(...findMdxFiles(fullPath));
    } else if (item.endsWith(".mdx") || item.endsWith(".md")) {
      files.push(fullPath);
    }
  }

  return files;
}

// Main function to generate search index
function generateSearchIndex(): void {
  const contentDir = path.join(process.cwd(), "content");

  if (!fs.existsSync(contentDir)) {
    console.warn("Content directory not found:", contentDir);
    return;
  }

  const mdxFiles = findMdxFiles(contentDir);
  const searchIndex: SearchIndex = {};

  console.log(`Found ${mdxFiles.length} markdown files`);

  for (const filePath of mdxFiles) {
    try {
      const fileContent = fs.readFileSync(filePath, "utf-8");
      const { data: frontmatter, content } = matter(fileContent);

      const { url, sectionTitle, sectionId } = getUrlAndSection(filePath, contentDir);
      const cleanContent = cleanMarkdownContent(content);

      // Use frontmatter title if available, otherwise derive from filename
      const title = frontmatter.title || path.basename(filePath, ".mdx").replace(/-/g, " ");

      searchIndex[url] = {
        title,
        content: cleanContent,
        sectionTitle,
        sectionId,
      };

      console.log(`Indexed: ${url} -> "${title}" (${sectionTitle})`);
    } catch (error) {
      console.error(`Error processing ${filePath}:`, error);
    }
  }

  // Generate TypeScript module
  const outputPath = path.join(process.cwd(), "src", "data", "searchIndex.ts");
  const outputDir = path.dirname(outputPath);

  // Ensure output directory exists
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const tsContent = `// This file is auto-generated. Do not edit manually.
// Generated on: ${new Date().toISOString()}

export interface SearchIndexEntry {
  title: string;
  content: string;
  sectionTitle: string;
  sectionId: string;
}

export type SearchIndex = Record<string, SearchIndexEntry>;

export const searchIndex: SearchIndex = ${JSON.stringify(searchIndex, null, 2)};

export default searchIndex;
`;

  fs.writeFileSync(outputPath, tsContent);
  console.log(`\nSearch index generated successfully!`);
  console.log(`Output: ${outputPath}`);
  console.log(`Indexed ${Object.keys(searchIndex).length} pages`);
}

// Run if called directly
if (require.main === module) {
  generateSearchIndex();
}

export { generateSearchIndex };
