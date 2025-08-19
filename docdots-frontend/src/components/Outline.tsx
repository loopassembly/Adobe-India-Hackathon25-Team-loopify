/* src/components/Outline.tsx
   Collapsible tree grouped by H1 → H4. Highlights the current page.
*/
import _, { useMemo, useState, Fragment } from "react";

type Item = { level: string; text: string; page: number };
type Props = {
  outline: Item[];
  onJump: (p: number) => void;
  currentPage?: number;
};

// ----- helpers to build a tree from the flat outline -----
type TreeNode = {
  id: string;
  depth: number; // 1..4 for H1..H4
  text: string;
  page: number;
  children: TreeNode[];
};

function lvlToDepth(lvl: string): number {
  const n = parseInt(lvl.replace(/[^\d]/g, ""), 10);
  if (!Number.isFinite(n) || n < 1) return 1;
  return Math.min(4, Math.max(1, n));
}

function buildTree(items: Item[]): TreeNode[] {
  const root: TreeNode = { id: "root", depth: 0, text: "root", page: -1, children: [] };
  const stack: TreeNode[] = [root];

  items.forEach((it, idx) => {
    const depth = lvlToDepth(it.level);
    const node: TreeNode = {
      id: `${idx}-${it.page}`,
      depth,
      text: it.text,
      page: it.page,
      children: [],
    };

    while (stack.length > 0 && stack[stack.length - 1].depth >= depth) {
      stack.pop();
    }
    const parent = stack[stack.length - 1] || root;
    parent.children.push(node);
    stack.push(node);
  });

  return root.children;
}

// simple chevron
function Chevron({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 20 20"
      width="14"
      height="14"
      className={`shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
      aria-hidden
    >
      <path fill="currentColor" d="M7 5l6 5l-6 5z" />
    </svg>
  );
}

export default function Outline({ outline, onJump, currentPage = -1 }: Props) {
  const tree = useMemo(() => buildTree(outline || []), [outline]);
  // track which H1/H2 nodes are expanded
  const [open, setOpen] = useState<Record<string, boolean>>({});

  // Reusable arrowhead marker for connectors
  const ArrowDefs = (
    <svg width="0" height="0" className="absolute">
      <defs>
        <marker id="dd-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 z" fill="currentColor" />
        </marker>
      </defs>
    </svg>
  );

  if (!outline || outline.length === 0) {
    return <div className="text-sm text-slate-500 p-2">No outline detected.</div>;
  }

  function Node({
    node,
    depth,
  }: {
    node: TreeNode;
    depth: number;
  }) {
    const hasKids = node.children && node.children.length > 0;
    const showToggle = hasKids && node.depth === 1; // only show chevron for H1
    const isOpen = open[node.id] ?? (node.depth === 1); // default open only H1
    const active = node.page === currentPage;

    const padd = depth * 12; // px; avoid Tailwind dynamic class issues

    return (
      <Fragment>
        <button
          onClick={() => onJump(node.page)}
          title={`Go to p${node.page + 1}`}
          className={`group w-full text-left rounded-lg border px-2 py-1 text-sm flex items-center gap-2 transition
            ${active ? "border-indigo-300 bg-indigo-50 text-indigo-800" : "border-transparent hover:bg-slate-50"}`}
          style={{ paddingLeft: padd }}
        >
          {showToggle ? (
            <span
              onClick={(e) => {
                e.stopPropagation();
                setOpen((s) => ({ ...s, [node.id]: !isOpen }));
              }}
              className="text-slate-500 hover:text-slate-700"
              aria-label={isOpen ? "Collapse" : "Expand"}
            >
              <Chevron open={isOpen} />
            </span>
          ) : (
            <span className="w-[14px]" />
          )}
        
          {/* Dotted elbow connector with arrow for nested nodes */}
          {depth > 0 && (
            <span className="text-slate-300 -ml-1">
              <svg width="18" height="14" viewBox="0 0 18 14" aria-hidden>
                <line
                  x1="0"
                  y1="7"
                  x2="14"
                  y2="7"
                  stroke="currentColor"
                  strokeDasharray="2 3"
                  strokeWidth="1.5"
                  markerEnd="url(#dd-arrow)"
                />
              </svg>
            </span>
          )}
        
          <span
            className={`inline-flex h-6 min-w-[34px] items-center justify-center rounded-md border px-2 text-[11px] font-semibold
              ${active ? "border-indigo-300 text-indigo-700 bg-white" : "border-slate-200 text-slate-600 bg-white"}
            `}
          >
            {`H${node.depth}`}
          </span>
        
          <span className="truncate">{node.text}</span>
        
          <span className="ml-auto text-[10px] text-slate-500">p{node.page + 1}</span>
        </button>

        {hasKids && isOpen && (
          <div className="space-y-1 pl-4">
            {node.children.map((c) => (
              <Node key={c.id} node={c} depth={depth + 1} />)
            )}
          </div>
        )}
      </Fragment>
    );
  }

  return (
    <div className="space-y-1 relative">
      {ArrowDefs}
      {tree.map((n) => (
        <Node key={n.id} node={n} depth={0} />
      ))}
    </div>
  );
}