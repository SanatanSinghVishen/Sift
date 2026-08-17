"use client";

import { useState } from "react";

interface CodeBlockProps {
  code: string;
  language: string;
  filename?: string;
}

export function CodeBlock({ code, language, filename }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="group relative rounded-xl border border-white/[0.08] bg-[#0d1117] overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/[0.06] bg-white/[0.02]">
        <div className="flex items-center gap-2">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-[#ff5f57]/80" />
            <div className="w-3 h-3 rounded-full bg-[#febc2e]/80" />
            <div className="w-3 h-3 rounded-full bg-[#28c840]/80" />
          </div>
          {filename && (
            <span className="text-xs text-muted-foreground font-mono ml-2">
              {filename}
            </span>
          )}
        </div>
        <button
          onClick={handleCopy}
          className="text-xs px-3 py-1 rounded-md bg-white/[0.06] hover:bg-white/[0.12] text-muted-foreground hover:text-foreground transition-all duration-200 font-mono cursor-pointer"
        >
          {copied ? "✓ Copied" : "Copy"}
        </button>
      </div>

      {/* Code content */}
      <pre className="p-4 overflow-x-auto text-sm leading-relaxed">
        <code className={`language-${language} text-slate-300 font-mono`}>
          {code}
        </code>
      </pre>
    </div>
  );
}
