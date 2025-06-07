import { getContentByPath, parseTableOfContents } from "@/lib/mdx";
import { notFound } from "next/navigation";
import MainLayout from "@/components/MainLayout";
import { MDXRemote } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// Custom MDX components
const components = {
  pre: (props) => <pre {...props} className="code-block" />,
  code: (props) => {
    const { children, className, ...rest } = props;
    return (
      <code className={className} {...rest}>
        {children}
      </code>
    );
  },
};

export default async function Page({ params }) {
  const { section, slug } = params;

  // Get the content
  const contentData = getContentByPath(section, slug);

  // If content not found, return 404
  if (!contentData) {
    return notFound();
  }

  // Parse table of contents
  const tocItems = parseTableOfContents(contentData.content);

  return (
    <MainLayout tocItems={tocItems}>
      <h1 className="page-title">{contentData.frontMatter.title}</h1>
      <div className="mdx-content">
        <MDXRemote
          source={contentData.content}
          components={components}
          options={{
            mdxOptions: {
              remarkPlugins: [remarkGfm, remarkMath],
              rehypePlugins: [rehypeHighlight, rehypeKatex],
            },
          }}
        />
      </div>
    </MainLayout>
  );
}
