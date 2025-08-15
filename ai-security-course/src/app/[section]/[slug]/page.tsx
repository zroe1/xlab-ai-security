import { getContentByPath, parseTableOfContents, getAllContentPaths } from "@/lib/mdx";
import MainLayout from "@/components/MainLayout";
import ExerciseButtons from "@/components/ExerciseButtons";
import ThemeImage from "@/components/ThemeImage";
import Dropdown from "@/components/Dropdown";
import OrganizationCard from "@/components/OrganizationCard";
import NextPageButton from "@/components/NextPageButton";
import FeedbackButton from "@/components/FeedbackButton";
import AdversarialScalingExplorer from "@/components/AdversarialScalingExplorer";
import { MDXRemote } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeCitation from "rehype-citation";
import React from "react";
import { navigationItems, type SubItem } from "../../../data/navigation";

// Helper function to extract all routes from navigation
function getAllNavigationRoutes() {
  const routes: { section: string; slug: string }[] = [];

  // Helper function to extract links recursively
  function extractLinks(items: SubItem[]): void {
    for (const item of items) {
      if (item.type === "link" && item.href !== "/") {
        // Parse the href to get section and slug
        const href = item.href.startsWith("/") ? item.href.slice(1) : item.href;
        const parts = href.split("/");
        if (parts.length === 2) {
          routes.push({
            section: parts[0],
            slug: parts[1],
          });
        }
      } else if (item.type === "folder") {
        extractLinks(item.items);
      }
    }
  }

  for (const section of navigationItems) {
    extractLinks(section.items);
  }

  return routes;
}

// Generate static params for all content pages and navigation routes
export async function generateStaticParams() {
  // Get paths from actual content files
  const contentPaths = getAllContentPaths();

  // Get all routes from navigation
  const navRoutes = getAllNavigationRoutes();

  // Combine and deduplicate
  const allRoutes = [...contentPaths];

  for (const navRoute of navRoutes) {
    // Only add if not already in contentPaths
    const exists = contentPaths.some(
      (cp) => cp.section === navRoute.section && cp.slug === navRoute.slug
    );
    if (!exists) {
      allRoutes.push(navRoute);
    }
  }

  return allRoutes.map((path) => ({
    section: path.section,
    slug: path.slug,
  }));
}

// Helper function to generate ID from text (matches the logic in parseTableOfContents)
const generateId = (text: string): string => {
  return text
    .toString()
    .toLowerCase()
    .replace(/\s+/g, "-")
    .replace(/[^\w-]/g, "");
};

// Custom heading components that add id attributes
const createHeading = (level: number) => {
  const Component = ({ children, ...props }: React.HTMLAttributes<HTMLHeadingElement>) => {
    const text =
      typeof children === "string" ? children : React.Children.toArray(children).join("");
    const id = generateId(text);

    return React.createElement(`h${level}`, { ...props, id }, children);
  };
  Component.displayName = `Heading${level}`;
  return Component;
};

interface PageProps {
  params: Promise<{
    section: string;
    slug: string;
  }>;
}

// Coming Soon component
const ComingSoon = () => {
  return (
    <MainLayout tocItems={[{ id: "coming-soon", text: "Coming Soon" }]}>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "450px",
          textAlign: "center",
          padding: "2rem",
        }}>
        <h1
          style={{
            fontSize: "2.2rem",
            fontWeight: "700",
            marginBottom: "0.3rem",
            color: "var(--text-primary)",
          }}>
          Coming Soon
        </h1>
        <p
          style={{
            fontSize: "1.0rem",
            color: "var(--text-secondary)",
            maxWidth: "600px",
            lineHeight: "1.6",
          }}>
          XLab&#39;s AI Security Guide is still a work in progress. Check back later when we have
          completed this page.
        </p>
      </div>
    </MainLayout>
  );
};

export default async function Page({ params }: PageProps) {
  const { section, slug } = await params;

  // Get the content
  const contentData = getContentByPath(section, slug);

  // If content not found, show Coming Soon page
  if (!contentData) {
    return <ComingSoon />;
  }

  // Parse table of contents
  const tocItems = parseTableOfContents(contentData.content);

  // Custom MDX components with section and slug context
  const components = {
    pre: (props: React.HTMLAttributes<HTMLPreElement>) => <pre {...props} className="code-block" />,
    code: (props: React.HTMLAttributes<HTMLElement>) => {
      const { children, className, ...rest } = props;
      return (
        <code className={className} {...rest}>
          {children}
        </code>
      );
    },
    h1: createHeading(1),
    h2: createHeading(2),
    h3: createHeading(3),
    h4: createHeading(4),
    h5: createHeading(5),
    h6: createHeading(6),
    ExerciseButtons,
    ThemeImage,
    Dropdown,
    OrganizationCard,
    NextPageButton,
    FeedbackButton,
    AdversarialScalingExplorer,
  };

  return (
    <MainLayout tocItems={tocItems}>
      <h1 className="page-title">{contentData.frontMatter.title as string}</h1>
      <div className="mdx-content">
        <MDXRemote
          source={contentData.content}
          components={components}
          options={{
            mdxOptions: {
              remarkPlugins: [remarkGfm, remarkMath],
              rehypePlugins: [
                rehypeHighlight,
                rehypeKatex,
                [
                  rehypeCitation,
                  { bibliography: "references.bib", csl: "apa", linkCitations: true },
                ],
              ],
            },
          }}
        />
      </div>
    </MainLayout>
  );
}
