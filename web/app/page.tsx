"use client";

import { useState } from "react";
import { SiftLogo } from "@/components/sift-logo";
import { Hero3DStack } from "@/components/hero-3d-stack";
import { CodeBlock } from "@/components/code-block";
import { BenchmarkCard } from "@/components/benchmark-card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// =============================================================================
// Code Snippets
// =============================================================================

const CODE_SNIPPETS = {
  python: `from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="sift-1b",
    messages=[
        {"role": "system", "content": "You are a function calling agent. Output only valid JSON tool calls."},
        {"role": "user", "content": "Schedule team sync for tomorrow 10 AM"}
    ],
    tools=[{
        "type": "function",
        "function": {
            "name": "create_event",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "time": {"type": "string"}
                },
                "required": ["title", "time"]
            }
        }
    }]
)
print(response.choices[0].message.tool_calls)`,

  typescript: `import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: "none",
});

const completion = await client.chat.completions.create({
  model: "sift-1b",
  messages: [
    { role: "system", content: "You are a function calling agent." },
    { role: "user", content: "Query active user count for last 24h" },
  ],
  tools: [{
    type: "function",
    function: {
      name: "get_active_users",
      parameters: {
        type: "object",
        properties: { hours: { type: "number" } },
        required: ["hours"],
      },
    },
  }],
});

console.log(completion.choices[0].message.tool_calls);`,

  curl: `curl -X POST http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "sift-1b",
    "messages": [
      {"role": "system", "content": "You are a function calling agent."},
      {"role": "user", "content": "Get weather in Seattle"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          }
        }
      }
    }]
  }'`,

  docker: `# Run the OpenAI-compatible FastAPI server locally
docker run -d -p 8000:8000 --gpus all \\
  ghcr.io/sanatansinghvishen/sift-1b:latest

# Or pull GGUF for Ollama
ollama run SanatanSinghVishen/sift-1b`,
};

const CLI_COMMANDS = [
  { label: "Ollama", cmd: "ollama run SanatanSinghVishen/sift-1b" },
  { label: "Docker", cmd: "docker run -p 8000:8000 --gpus all ghcr.io/sanatansinghvishen/sift-1b" },
];

// =============================================================================
// Page Component
// =============================================================================

export default function HomePage() {
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const copyCommand = async (cmd: string) => {
    await navigator.clipboard.writeText(cmd);
    setCopiedCmd(cmd);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  return (
    <main className="flex flex-col min-h-screen bg-[#0A0A0A] text-[#FFFFFF] font-sans antialiased selection:bg-white selection:text-black">
      {/* ================================================================= */}
      {/*  NAVBAR                                                           */}
      {/* ================================================================= */}
      <nav className="sticky top-0 z-50 border-b border-[#27272A]/80 bg-[#0A0A0A]/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-8 h-8 rounded-full border border-white/20 bg-white/5">
              <SiftLogo className="w-5 h-5 text-white" />
            </div>
            <span className="font-medium text-base tracking-tight text-white">Sift</span>
          </div>

          {/* Center Links */}
          <div className="hidden md:flex items-center gap-9 text-sm text-[#A1A1AA] font-normal">
            <a href="#benchmarks" className="hover:text-white transition-colors">
              Products
            </a>
            <a href="#architecture" className="hover:text-white transition-colors">
              Solutions
            </a>
            <a href="#integration" className="hover:text-white transition-colors">
              Developers
            </a>
            <a href="#specifications" className="hover:text-white transition-colors">
              Pricing
            </a>
          </div>

          {/* Right Actions */}
          <div className="flex items-center gap-5">
            <a
              href="https://github.com/SanatanSinghVishen/Sift"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-[#A1A1AA] hover:text-white transition-colors font-medium hidden sm:inline-block"
            >
              Sign In
            </a>
            <a
              href="https://huggingface.co/SanatanSinghVishen/sift-1b"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center justify-center px-5 py-2 rounded-full border border-[#27272A] bg-[#151515] text-xs font-medium text-white hover:border-white/40 hover:bg-[#1a1a1a] transition-all cursor-pointer shadow-sm"
            >
              Contact Sales
            </a>
          </div>
        </div>
      </nav>

      {/* ================================================================= */}
      {/*  HERO SECTION (Platform Architecture Style)                       */}
      {/* ================================================================= */}
      <section className="relative overflow-hidden pt-12 md:pt-20 pb-20 md:pb-28">
        {/* Subtle Ambient Radial Lighting */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] bg-white/[0.015] rounded-full blur-[140px] pointer-events-none" />

        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center min-h-[580px]">
            
            {/* Left Column: Hero Content */}
            <div className="lg:col-span-6 flex flex-col justify-center z-10">
              {/* Display Lg Heading */}
              <h1 className="text-5xl sm:text-6xl md:text-[68px] leading-[1.08] font-medium tracking-tight text-white mb-6">
                One ecosystem.<br />
                Infinite<br />
                dimensions.
              </h1>

              {/* Body Md Subtitle */}
              <p className="text-base sm:text-lg text-[#A1A1AA] leading-relaxed max-w-xl mb-10 font-normal">
                Unify your workflow orchestration, schema routing, and local function execution in a single, high-performance spatial interface.
              </p>

              {/* Action Buttons (Pill Radius: 9999px) */}
              <div className="flex flex-wrap items-center gap-4">
                <a
                  href="#integration"
                  className="inline-flex items-center justify-center gap-2 px-7 py-3.5 rounded-full bg-white text-[#080808] font-medium text-sm hover:bg-neutral-200 transition-all cursor-pointer shadow-lg shadow-white/5"
                >
                  Start Creating →
                </a>
                <a
                  href="#benchmarks"
                  className="inline-flex items-center justify-center px-7 py-3.5 rounded-full border border-[#27272A] bg-[#080808] text-white font-medium text-sm hover:border-white/30 hover:bg-[#151515] transition-all cursor-pointer"
                >
                  Read Documentation
                </a>
              </div>
            </div>

            {/* Right Column: Interactive 3D Stacked Layers (Cursor-Tracking Parallax) */}
            <div className="lg:col-span-6 flex items-center justify-center lg:justify-end z-10">
              <Hero3DStack />
            </div>

          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/*  CLI QUICK-START                                                  */}
      {/* ================================================================= */}
      <section className="py-8 border-y border-[#27272A]/50 bg-[#080808]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {CLI_COMMANDS.map(({ label, cmd }) => (
              <div
                key={label}
                className="flex items-center justify-between gap-4 rounded-full border border-[#27272A] bg-[#151515] px-6 py-3 hover:border-white/20 transition-colors"
              >
                <div className="flex items-center gap-3 overflow-hidden">
                  <span className="text-xs font-mono text-[#A1A1AA] uppercase tracking-wider shrink-0">
                    {label}
                  </span>
                  <code className="text-xs sm:text-sm font-mono text-white/90 truncate">
                    $ {cmd}
                  </code>
                </div>
                <button
                  onClick={() => copyCommand(cmd)}
                  className="text-xs px-4 py-1.5 rounded-full border border-white/10 bg-white/5 hover:bg-white/10 text-white font-mono shrink-0 transition-all cursor-pointer"
                >
                  {copiedCmd === cmd ? "✓ Copied" : "Copy"}
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/*  BENCHMARKS SECTION                                               */}
      {/* ================================================================= */}
      <section id="benchmarks" className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="mb-14">
            <Badge variant="outline" className="mb-3 font-mono text-xs text-[#A1A1AA] border-[#27272A] rounded-full px-3 py-1 bg-[#151515]">
              Empirical Evaluation
            </Badge>
            <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-white mb-4">
              Benchmarks & Specifications
            </h2>
            <p className="text-base text-[#A1A1AA] max-w-2xl">
              Evaluated across 500 unseen schemas on holdout test sets. Measured on local hardware with zero cloud API dependencies.
            </p>
          </div>

          {/* 4 Core Metric Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <BenchmarkCard
              label="Tool Selection Acc"
              value="100.0%"
              target="Base Model: 70.0%"
              icon="🎯"
            />
            <BenchmarkCard
              label="Param Extraction"
              value="88.0%"
              target="Base Model: 34.0%"
              icon="📋"
            />
            <BenchmarkCard
              label="Clean JSON Rate"
              value="100.0%"
              target="0% Markdown / Fluff"
              icon="🧼"
            />
            <BenchmarkCard
              label="Inference Latency"
              value="1714 ms"
              target="25% Faster than Base"
              icon="⚡"
            />
          </div>

          {/* 3-Tier Benchmark Table */}
          <div className="rounded-[32px] border border-[#27272A] bg-[#151515] overflow-hidden mb-8">
            <div className="px-8 py-5 border-b border-[#27272A] flex items-center justify-between">
              <div>
                <h3 className="font-medium text-base text-white">3-Tier Empirical Benchmark Progression</h3>
                <p className="text-xs text-[#A1A1AA] font-mono mt-0.5">Base Qwen → SFT Checkpoint → Sift-1B (DPO Step 750)</p>
              </div>
              <Badge variant="outline" className="text-white border-white/20 text-xs rounded-full bg-white/5">
                Golden Checkpoint
              </Badge>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-[#27272A] text-[#A1A1AA] text-xs font-mono uppercase bg-[#0d0d0d]">
                    <th className="px-8 py-4">Metric</th>
                    <th className="px-8 py-4 text-right">Qwen-1.5B (Base)</th>
                    <th className="px-8 py-4 text-right">Sift-1B (SFT)</th>
                    <th className="px-8 py-4 text-right text-white font-semibold">Sift-1B (DPO - Ours)</th>
                    <th className="px-8 py-4 text-right">Delta vs Base</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#27272A]/60 font-mono text-sm">
                  {[
                    ["Tool Selection Accuracy", "70.0%", "98.0%", "100.0% 🏆", "+30.0%"],
                    ["Parameter Extraction", "34.0%", "80.0%", "88.0% 🏆", "+54.0%"],
                    ["JSON Parse Rate", "96.0%", "98.0%", "100.0% 🏆", "+4.0%"],
                    ["Zero Markdown / Fluff", "76.0%", "100.0%", "100.0% 🏆", "+24.0% (Flawless)"],
                    ["Zero Hallucinations", "100.0%", "100.0%", "100.0% 🏆", "0% Hallucinations"],
                    ["Average Latency (TTFT)", "2277 ms", "1734 ms", "1714 ms ⚡", "25% Faster"],
                  ].map(([metric, base, sft, dpo, delta], idx) => (
                    <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                      <td className="px-8 py-4 font-sans font-normal text-white">{metric}</td>
                      <td className="px-8 py-4 text-right text-[#A1A1AA]">{base}</td>
                      <td className="px-8 py-4 text-right text-white/80">{sft}</td>
                      <td className="px-8 py-4 text-right text-white font-bold">{dpo}</td>
                      <td className="px-8 py-4 text-right text-[#A1A1AA] text-xs">{delta}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Training Configuration */}
          <div id="specifications" className="rounded-[32px] border border-[#27272A] bg-[#151515] overflow-hidden">
            <div className="px-8 py-5 border-b border-[#27272A]">
              <h3 className="font-medium text-base text-white">System & Training Configuration</h3>
            </div>
            <div className="divide-y divide-[#27272A]/50">
              {[
                ["Base Architecture", "Qwen2.5-1.5B-Instruct"],
                ["Quantization", "QLoRA (4-bit NF4) / GGUF Q4_K_M"],
                ["Adapter Hyperparameters", "r=16, α=32, target_modules=[q,k,v,o,gate,up,down]"],
                ["SFT Dataset", "Salesforce/xlam-function-calling-60k (10,000 ChatML rows)"],
                ["DPO Preference Strategy", "Synthetically mutated negative pairs (syntax + args)"],
                ["Optimal Alignment Step", "Step 750 Golden Checkpoint (β=0.1, sigmoid loss)"],
                ["Hardware", "NVIDIA RTX 3050 (4 GB VRAM) & Tesla T4"],
                ["Export Formats", "GGUF Q4_K_M + SafeTensors Fused Weights"],
              ].map(([key, val]) => (
                <div
                  key={key}
                  className="flex flex-col sm:flex-row sm:items-center px-8 py-4 text-sm hover:bg-white/[0.02] transition-colors gap-1 sm:gap-0"
                >
                  <span className="w-64 text-[#A1A1AA] font-normal shrink-0">
                    {key}
                  </span>
                  <span className="font-mono text-white/90 text-xs sm:text-sm">{val}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/*  INTEGRATION CODE SNIPPETS                                        */}
      {/* ================================================================= */}
      <section id="integration" className="py-24 border-t border-[#27272A]/50 bg-[#080808]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="mb-14">
            <Badge variant="outline" className="mb-3 font-mono text-xs text-[#A1A1AA] border-[#27272A] rounded-full px-3 py-1 bg-[#151515]">
              Developer Integration
            </Badge>
            <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-white mb-4">
              Drop-In Cloud API Replacement
            </h2>
            <p className="text-base text-[#A1A1AA] max-w-xl">
              Compatible with the OpenAI SDK specification. Point your existing client to localhost and run offline with zero per-token billing.
            </p>
          </div>

          <Tabs defaultValue="python" className="w-full">
            <TabsList className="w-full max-w-md grid grid-cols-4 mb-6 rounded-full border border-[#27272A] bg-[#151515] p-1">
              <TabsTrigger value="python" className="font-mono text-xs rounded-full data-[state=active]:bg-white data-[state=active]:text-black">
                Python
              </TabsTrigger>
              <TabsTrigger value="typescript" className="font-mono text-xs rounded-full data-[state=active]:bg-white data-[state=active]:text-black">
                TypeScript
              </TabsTrigger>
              <TabsTrigger value="curl" className="font-mono text-xs rounded-full data-[state=active]:bg-white data-[state=active]:text-black">
                cURL
              </TabsTrigger>
              <TabsTrigger value="docker" className="font-mono text-xs rounded-full data-[state=active]:bg-white data-[state=active]:text-black">
                Docker
              </TabsTrigger>
            </TabsList>

            <div className="rounded-[32px] border border-[#27272A] bg-[#151515] overflow-hidden p-2">
              <TabsContent value="python" className="m-0">
                <CodeBlock code={CODE_SNIPPETS.python} language="python" filename="main.py" />
              </TabsContent>
              <TabsContent value="typescript" className="m-0">
                <CodeBlock code={CODE_SNIPPETS.typescript} language="typescript" filename="index.ts" />
              </TabsContent>
              <TabsContent value="curl" className="m-0">
                <CodeBlock code={CODE_SNIPPETS.curl} language="bash" filename="terminal" />
              </TabsContent>
              <TabsContent value="docker" className="m-0">
                <CodeBlock code={CODE_SNIPPETS.docker} language="bash" filename="terminal" />
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </section>

      {/* ================================================================= */}
      {/*  ARCHITECTURE PIPELINE                                            */}
      {/* ================================================================= */}
      <section id="architecture" className="py-24 border-t border-[#27272A]/50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="mb-14">
            <Badge variant="outline" className="mb-3 font-mono text-xs text-[#A1A1AA] border-[#27272A] rounded-full px-3 py-1 bg-[#151515]">
              System Topology
            </Badge>
            <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-white mb-4">
              SLM → Router → Execution Pipeline
            </h2>
            <p className="text-base text-[#A1A1AA] max-w-xl">
              Sift acts as an edge-native load balancer. It processes structured parameters instantly and routes execution without latency spikes.
            </p>
          </div>

          <div className="rounded-[32px] border border-[#27272A] bg-[#151515] p-8 md:p-12">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
              {/* Node 1 */}
              <div className="rounded-[24px] border border-[#27272A] bg-[#0A0A0A] p-6 text-center">
                <div className="w-10 h-10 rounded-full border border-white/20 bg-white/5 flex items-center justify-center mx-auto mb-4 text-white">
                  1
                </div>
                <h4 className="font-medium text-base text-white mb-1">User Input</h4>
                <p className="text-xs text-[#A1A1AA] font-mono">Unstructured Natural Prompt</p>
              </div>

              {/* Node 2 */}
              <div className="rounded-[24px] border border-white/30 bg-[#0A0A0A] p-6 text-center shadow-[0_0_24px_rgba(255,255,255,0.03)]">
                <div className="w-10 h-10 rounded-full border border-white bg-white text-black flex items-center justify-center mx-auto mb-4 font-bold">
                  2
                </div>
                <h4 className="font-medium text-base text-white mb-1">Sift-1B Router</h4>
                <p className="text-xs text-[#A1A1AA] font-mono">~30ms Deterministic JSON</p>
              </div>

              {/* Node 3 */}
              <div className="rounded-[24px] border border-[#27272A] bg-[#0A0A0A] p-6 text-center">
                <div className="w-10 h-10 rounded-full border border-white/20 bg-white/5 flex items-center justify-center mx-auto mb-4 text-white">
                  3
                </div>
                <h4 className="font-medium text-base text-white mb-1">Tool Execution</h4>
                <p className="text-xs text-[#A1A1AA] font-mono">API / Backend / Database</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/*  FOOTER                                                           */}
      {/* ================================================================= */}
      <footer className="py-14 border-t border-[#27272A]/50 bg-[#080808]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center w-7 h-7 rounded-full border border-white/20 bg-white/5">
                <SiftLogo className="w-4 h-4 text-white" />
              </div>
              <span className="text-sm font-medium text-white">Sift-1B</span>
              <span className="text-xs text-[#A1A1AA] font-mono ml-2">
                Built by Sanatan Singh Vishen
              </span>
            </div>
            <div className="flex items-center gap-8 text-sm text-[#A1A1AA]">
              <a
                href="https://huggingface.co/SanatanSinghVishen/sift-1b"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-white transition-colors"
              >
                Hugging Face
              </a>
              <a
                href="https://github.com/SanatanSinghVishen/Sift"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-white transition-colors"
              >
                GitHub
              </a>
              <span className="text-xs font-mono">MIT License</span>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}
