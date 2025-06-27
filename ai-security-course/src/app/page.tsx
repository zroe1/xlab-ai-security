import { getContentByPath, parseTableOfContents } from "@/lib/mdx";
import { notFound } from "next/navigation";
import MainLayout from "@/components/MainLayout";
import ExerciseButtons from "@/components/ExerciseButtons";
import { MDXRemote } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeCitation from "rehype-citation";
import React from "react";

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

// Custom MDX components
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
};

export default function Home() {
  // Load the installation content from MDX
  const contentData = getContentByPath("", "installation");

  // If content not found, return 404
  if (!contentData) {
    return notFound();
  }

  // Parse table of contents
  const tocItems = parseTableOfContents(contentData.content);

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
