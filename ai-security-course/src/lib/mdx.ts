import fs from "fs";
import path from "path";
import matter from "gray-matter";

// The content directory with all MDX files
const contentDirectory = path.join(process.cwd(), "content");

// Type definitions
interface ContentPath {
  section: string;
  slug: string;
}

interface ContentData {
  frontMatter: Record<string, unknown>;
  content: string;
  section: string;
  slug: string;
}

interface TocItem {
  level: number;
  text: string;
  id: string;
}

/**
 * Get all available paths from the content directory
 */
export function getAllContentPaths(): ContentPath[] {
  console.log("Content directory:", contentDirectory);
  const sections = fs.readdirSync(contentDirectory);
  console.log("Sections found:", sections);

  // Gather all paths across all sections
  const paths: ContentPath[] = [];

  for (const section of sections) {
    const sectionPath = path.join(contentDirectory, section);
    if (fs.statSync(sectionPath).isDirectory()) {
      const files = fs.readdirSync(sectionPath);
      console.log(`Files in section ${section}:`, files);

      for (const file of files) {
        if (file.endsWith(".mdx") || file.endsWith(".md")) {
          paths.push({
            section,
            slug: file.replace(/\.(mdx|md)$/, ""),
          });
        }
      }
    }
  }

  console.log("Paths found:", paths);
  return paths;
}

/**
 * Get content for a specific path
 */
export function getContentByPath(section: string, slug: string): ContentData | null {
  console.log("Looking for content:", section, slug);

  let fullPath: string;

  // If section is empty, look in the root content directory
  if (!section || section === "") {
    fullPath = path.join(contentDirectory, `${slug}.mdx`);
  } else {
    fullPath = path.join(contentDirectory, section, `${slug}.mdx`);
  }

  console.log("Checking path:", fullPath);
  console.log("Path exists?", fs.existsSync(fullPath));

  if (!fs.existsSync(fullPath)) {
    // Try .md extension
    if (!section || section === "") {
      fullPath = path.join(contentDirectory, `${slug}.md`);
    } else {
      fullPath = path.join(contentDirectory, section, `${slug}.md`);
    }
    console.log("Checking alternate path:", fullPath);
    console.log("Alternate path exists?", fs.existsSync(fullPath));

    if (!fs.existsSync(fullPath)) {
      console.log("Content not found!");
      return null;
    }
  }

  console.log("Reading file:", fullPath);
  const fileContents = fs.readFileSync(fullPath, "utf8");
  const { data, content } = matter(fileContents);
  console.log("Frontmatter:", data);

  return {
    frontMatter: data,
    content,
    section: section || "",
    slug,
  };
}

/**
 * Parse headings from markdown content for table of contents
 */
export function parseTableOfContents(content: string): TocItem[] {
  const headings: TocItem[] = [];
  const lines = content.split("\n");

  for (const line of lines) {
    if (line.startsWith("#")) {
      const match = line.match(/^#+/);
      if (match) {
        const level = match[0].length;
        const text = line.replace(/^#+\s+/, "");
        const id = text
          .toLowerCase()
          .replace(/\s+/g, "-")
          .replace(/[^\w-]/g, "");

        headings.push({ level, text, id });
      }
    }
  }

  return headings;
}
