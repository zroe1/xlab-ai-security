declare module "@benchmark-urbanism/remark-bibtex" {
  import { Plugin } from "unified";

  interface RemarkBibtexOptions {
    bibtexFile: string;
  }

  const remarkBibtex: Plugin<[RemarkBibtexOptions?]>;
  export default remarkBibtex;
}
