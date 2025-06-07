import { getContentByPath, parseTableOfContents } from "../../../../lib/mdx";
// import { notFound } from "next/navigation";
// import MainLayout from "../../../components/MainLayout";

// import remarkGfm from "remark-gfm";
// import rehypeHighlight from "rehype-highlight";
// import remarkMath from "remark-math";
// import rehypeKatex from "rehype-katex";

// // Custom MDX components
// const components = {
//   pre: (props) => <pre {...props} className="code-block" />,
//   code: (props) => {
//     const { children, className, ...rest } = props;
//     return (
//       <code className={className} {...rest}>
//         {children}
//       </code>
//     );
//   },
// };

// export default async function Page({ params }) {
//   const { section, slug } = params;

//   console.log("Rendering dynamic page with:", section, slug);

//   // Get the content
//   const contentData = getContentByPath(section, slug);

//   // If content not found, return 404
//   if (!contentData) {
//     console.log("Content not found for:", section, slug);
//     return notFound();
//   }

//   // Parse table of contents
//   const tocItems = parseTableOfContents(contentData.content);

//   // Process the MDX content
//   try {
//     const { content } = await compileMDX({
//       source: contentData.content,
//       components,
//       options: {
//         mdxOptions: {
//           remarkPlugins: [remarkGfm, remarkMath],
//           rehypePlugins: [rehypeHighlight, rehypeKatex],
//         },
//       },
//     });

//     return (
//       <MainLayout tocItems={tocItems}>
//         <h1 className="page-title">{contentData.frontMatter.title}</h1>
//         <div className="mdx-content">{content}</div>
//       </MainLayout>
//     );
//   } catch (error) {
//     console.error("Error processing MDX:", error);

//     // Fallback to basic rendering if MDX processing fails
//     return (
//       <MainLayout tocItems={tocItems}>
//         <h1 className="page-title">{contentData.frontMatter.title}</h1>
//         <div>
//           <p>Error rendering content. Please check the console for details.</p>
//           <pre style={{ whiteSpace: "pre-wrap" }}>{contentData.content.slice(0, 500)}...</pre>
//         </div>
//       </MainLayout>
//     );
//   }
// }

// // This helps with debugging - log all params at build time
// export function generateStaticParams() {
//   console.log("Generating static params...");
//   return [];
// }

// import { getContentByPath, parseTableOfContentss } from "../../../lib/mdx";
import { notFound } from "next/navigation";
import MainLayout from "../../../components/MainLayout";
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

  console.log("Rendering dynamic page with:", section, slug);

  // Get the content
  const contentData = getContentByPath(section, slug);

  // If content not found, return 404
  if (!contentData) {
    console.log("Content not found for:", section, slug);
    return notFound();
  }

  // Parse table of contents
  const tocItems = parseTableOfContents(contentData.content);

  // Process the MDX content using MDXRemote directly (not compileMDX)
  try {
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
  } catch (error) {
    console.error("Error processing MDX:", error);

    // Fallback to basic rendering if MDX processing fails
    return (
      <MainLayout tocItems={tocItems}>
        <h1 className="page-title">{contentData.frontMatter.title}</h1>
        <div>
          <p>Error rendering content.</p>
          <div dangerouslySetInnerHTML={{ __html: `<pre>${error.message}</pre>` }} />
        </div>
      </MainLayout>
    );
  }
}
