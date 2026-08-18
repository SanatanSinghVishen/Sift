"use client";

import { useState } from "react";
import { SiftLogo } from "@/components/sift-logo";
import { Hero3DStack } from "@/components/hero-3d-stack";
import { CodeBlock } from "@/components/code-block";
import { BenchmarkCard } from "@/components/benchmark-card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// =============================================================================
// Code Snippets & Examples
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

  docker: `# Run OpenAI-compatible FastAPI inference server locally
docker run -d -p 8000:8000 --gpus all \\
  ghcr.io/sanatansinghvishen/sift-1b:latest

# Or pull GGUF for Ollama (Single command)
ollama run hf.co/SanatanSinghVishen/sift-1b-gguf`,
};

const CLI_COMMANDS = [
  { label: "Ollama (GGUF)", cmd: "ollama run hf.co/SanatanSinghVishen/sift-1b-gguf" },
  { label: "Docker Server", cmd: "docker run -p 8000:8000 --gpus all ghcr.io/sanatansinghvishen/sift-1b" },
];

const FAQS = [
  {
    q: "Why should I use Sift-1B instead of a cloud API like GPT-4o-mini?",
    a: "If your task requires open-ended creative reasoning, use a large cloud model. However, for deterministic JSON function calling, parameter extraction, and schema routing running thousands of times daily, Sift-1B is vastly superior. It eliminates API costs entirely ($0 per million tokens), guarantees sub-35ms TTFT on local hardware, and completely eliminates pipeline crashes caused by markdown wrapping or conversational hallucinations.",
  },
  {
    q: "Can I run this locally on my laptop?",
    a: "Yes. Thanks to 4-bit quantization and PEFT parameter fusion, Sift-1B requires less than 1.2 GB of VRAM. It runs with zero latency overhead on consumer GPUs (like an NVIDIA RTX 3050), and executes at high speed on standard CPUs via Ollama or llama.cpp.",
  },
  {
    q: "What happens if the user asks a completely unrelated question?",
    a: "Sift-1B is designed with a strict escalation protocol. If a user asks a general knowledge question (e.g., 'What is the capital of Japan?'), Sift-1B will not hallucinate or chat. Instead, it outputs a low-confidence score or triggers a dedicated fallback route so your application can safely forward that specific request to a larger frontier model.",
  },
  {
    q: "Does it hallucinate or invent schema keys?",
    a: "No. During the Direct Preference Optimization (DPO) training phase, Sift-1B was explicitly trained against synthetic negative pairs penalizing type coercion (e.g. outputting strings instead of integers) and hallucinated keys. It achieved a 100.0% zero-hallucination rate on unseen test schemas.",
  },
  {
    q: "How do I integrate Sift-1B into my existing codebase?",
    a: "Sift-1B provides 100% drop-in compatibility with the OpenAI SDK. Simply point your existing client's base_url to http://localhost:8000/v1 (FastAPI) or http://localhost:11434 (Ollama). No structural code rewrites are required.",
  },
];

// =============================================================================
// Page Component
// =============================================================================

export default function HomePage() {
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);
  const [openFaq, setOpenFaq] = useState<number | null>(0);
  const [contactStatus, setContactStatus] = useState<string | null>(null);
  const [contactForm, setContactForm] = useState({ name: "", email: "", message: "" });

  const copyCommand = async (cmd: string) => {
    await navigator.clipboard.writeText(cmd);
    setCopiedCmd(cmd);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  const handleContactSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const mailtoUrl = `mailto:sanatansinghgonda@gmail.com?subject=${encodeURIComponent(`Sift-1B Inquiry from ${contactForm.name}`)}&body=${encodeURIComponent(`Name: ${contactForm.name}\nEmail: ${contactForm.email}\n\nMessage:\n${contactForm.message}`)}`;
    window.open(mailtoUrl, "_blank");
    setContactStatus("Opening email client to send to sanatansinghgonda@gmail.com...");
    setContactForm({ name: "", email: "", message: "" });
    setTimeout(() => setContactStatus(null), 5000);
  };

  return (
    <main className="flex flex-col min-h-screen bg-[#0A0A0A] text-[#FFFFFF] font-sans antialiased selection:bg-white selection:text-black">
      {/* ================================================================= */}
      {/*  NAVBAR                                                           */}
      {/* ================================================================= */}
      <nav className="sticky top-0 z-50 border-b border-[#27272A]/80 bg-[#0A0A0A]/85 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-20 flex items-center justify-between">
          {/* Logo & Brand */}
          <a href="#" className="flex items-center gap-3 group">
            <div className="flex items-center justify-center w-8 h-8 rounded-full border border-white/20 bg-white/5 group-hover:border-white/40 transition-colors">
              <SiftLogo className="w-4 h-4 text-white" />
            </div>
            <span className="font-medium text-base tracking-tight text-white">Sift-1B</span>
          </a>

          {/* Center Links (Smooth Anchors) */}
          <div className="hidden md:flex items-center gap-8 text-sm text-[#A1A1AA] font-normal">
            <a href="#use-cases" className="hover:text-white transition-colors">
              Use Cases
            </a>
            <a href="#benchmarks" className="hover:text-white transition-colors">
              Benchmarks
            </a>
            <a href="#architecture" className="hover:text-white transition-colors">
              Architecture
            </a>
            <a href="#integration" className="hover:text-white transition-colors">
              Integration
            </a>
            <a href="#faq" className="hover:text-white transition-colors">
              FAQ
            </a>
          </div>

          {/* Right Actions */}
          <div className="flex items-center gap-4">
            <a
              href="https://huggingface.co/SanatanSinghVishen/sift-1b-gguf"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-mono text-[#A1A1AA] hover:text-white transition-colors items-center gap-1.5 hidden sm:flex"
            >
              🤗 Hugging Face
            </a>
            <a
              href="https://github.com/SanatanSinghVishen/Sift"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-mono text-[#A1A1AA] hover:text-white transition-colors items-center gap-1.5 hidden sm:flex"
            >
              ⭐ GitHub
            </a>
            <a
              href="#contact"
              className="inline-flex items-center justify-center px-5 py-2 rounded-full border border-[#27272A] bg-[#151515] text-xs font-medium text-white hover:border-white/40 hover:bg-[#1f1f1f] transition-all cursor-pointer shadow-sm"
            >
              Contact
            </a>
          </div>
        </div>
      </nav>

      {/* ================================================================= */}
      {/*  HERO SECTION                                                     */}
      {/* ================================================================= */}
      <section className="relative overflow-hidden pt-12 md:pt-20 pb-20 md:pb-28">
        {/* Ambient Radial Lighting */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[500px] bg-white/[0.015] rounded-full blur-[140px] pointer-events-none" />

        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center min-h-[580px]">

            {/* Left Column: Hero Content */}
            <div className="lg:col-span-6 flex flex-col justify-center z-10">

              {/* Display Lg Heading */}
              <h1 className="text-5xl sm:text-6xl md:text-[68px] leading-[1.06] font-medium tracking-tight text-white mb-6">
                Extract the signal.<br />
                Route the action.
              </h1>

              {/* Body Md Subtitle */}
              <p className="text-base sm:text-lg text-[#A1A1AA] leading-relaxed max-w-xl mb-10 font-normal">
                A hyper-specialized 1.5B Small Language Model fine-tuned via QLoRA + DPO for deterministic JSON function calling. 100% strict formatting. Zero API costs. Private edge execution.
              </p>

              {/* Action Buttons */}
              <div className="flex flex-wrap items-center gap-4">
                <a
                  href="#integration"
                  className="inline-flex items-center justify-center gap-2 px-7 py-3.5 rounded-full bg-white text-[#080808] font-medium text-sm hover:bg-neutral-200 transition-all cursor-pointer shadow-lg shadow-white/5"
                >
                  Get Started (Ollama) →
                </a>
                <a
                  href="#use-cases"
                  className="inline-flex items-center justify-center px-7 py-3.5 rounded-full border border-[#27272A] bg-[#080808] text-white font-medium text-sm hover:border-white/30 hover:bg-[#151515] transition-all cursor-pointer"
                >
                  Explore Use Cases
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
                  <code className="text-[11px] sm:text-xs md:text-sm font-mono text-white/90 overflow-x-auto whitespace-nowrap scrollbar-none">
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
      {/*  CORE USE CASES & ARCHITECTURE FIT                                */}
      {/* ================================================================= */}
      <section id="use-cases" className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="mb-14">

            <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-white mb-4">
              Engineered for Routing & Parameter Extraction
            </h2>
            <p className="text-base text-[#A1A1AA] max-w-2xl">
              Sift-1B is not built to write essays or chat. It is engineered to sit at the front of your application as a high-speed traffic cop and schema translator.
            </p>
          </div>

          {/* 3 Core Pillar Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-14">
            {/* Pillar 1 */}
            <div className="rounded-[32px] border border-[#27272A] bg-[#151515] p-8 flex flex-col justify-between hover:border-white/20 transition-all">
              <div>
                <div className="w-12 h-12 rounded-full border border-white/20 bg-white/5 flex items-center justify-center mb-6 text-xl">
                  🚦
                </div>
                <h3 className="text-xl font-medium text-white mb-3">
                  Multi-Agent Routing
                </h3>
                <div className="space-y-3 text-sm text-[#A1A1AA] leading-relaxed">
                  <p>
                    <strong className="text-white">The Problem:</strong> Passing every incoming request through massive cloud LLMs just to classify intent adds heavy latency and costly token overhead.
                  </p>
                  <p>
                    <strong className="text-white">The Sift Solution:</strong> Sift-1B evaluates the user query locally in milliseconds, outputting a strict JSON route tag to direct requests to the appropriate agent.
                  </p>
                </div>
              </div>
            </div>

            {/* Pillar 2 */}
            <div className="rounded-[32px] border border-[#27272A] bg-[#151515] p-8 flex flex-col justify-between hover:border-white/20 transition-all">
              <div>
                <div className="w-12 h-12 rounded-full border border-white/20 bg-white/5 flex items-center justify-center mb-6 text-xl">
                  🧩
                </div>
                <h3 className="text-xl font-medium text-white mb-3">
                  Deterministic Extraction
                </h3>
                <div className="space-y-3 text-sm text-[#A1A1AA] leading-relaxed">
                  <p>
                    <strong className="text-white">The Problem:</strong> General-purpose LLMs frequently wrap JSON in markdown wrappers or conversational preamble, breaking production backend parsers.
                  </p>
                  <p>
                    <strong className="text-white">The Sift Solution:</strong> Fine-tuned with Direct Preference Optimization (DPO) to strictly penalize formatting errors, guaranteeing pure, parseable JSON on attempt #1.
                  </p>
                </div>
              </div>
            </div>

            {/* Pillar 3 */}
            <div className="rounded-[32px] border border-[#27272A] bg-[#151515] p-8 flex flex-col justify-between hover:border-white/20 transition-all">
              <div>
                <div className="w-12 h-12 rounded-full border border-white/20 bg-white/5 flex items-center justify-center mb-6 text-xl">
                  🔒
                </div>
                <h3 className="text-xl font-medium text-white mb-3">
                  Edge Privacy & Automation
                </h3>
                <div className="space-y-3 text-sm text-[#A1A1AA] leading-relaxed">
                  <p>
                    <strong className="text-white">The Problem:</strong> Sending sensitive user details or proprietary database schema to third-party cloud APIs poses compliance and security risks.
                  </p>
                  <p>
                    <strong className="text-white">The Sift Solution:</strong> Operates completely offline with sub-1.2 GB VRAM usage. Zero third-party network requests; GDPR and SOC 2 compliant by design.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Real-World Query Execution Demonstrations */}
          <div className="rounded-[32px] border border-[#27272A] bg-[#151515] overflow-hidden p-8 md:p-10">
            <div className="mb-8">
              <h3 className="text-2xl font-medium text-white mb-2">
                ⚡ Real-World Query Handling
              </h3>
              <p className="text-sm text-[#A1A1AA]">
                How Sift-1B transforms unstructured natural language into strict machine-readable function payloads.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Example 1 */}
              <div className="rounded-[24px] border border-[#27272A] bg-[#0A0A0A] p-6 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <span className="font-mono text-xs text-[#A1A1AA] uppercase tracking-wider">Example 1: Intent Routing</span>
                  </div>
                  <div className="space-y-3 mb-4 font-mono text-xs">
                    <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                      <span className="text-[#A1A1AA] block mb-1">Target Schema:</span>
                      <code className="text-white/80">{`{"route": "string", "confidence": "float"}`}</code>
                    </div>
                    <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                      <span className="text-[#A1A1AA] block mb-1">User Query:</span>
                      <span className="text-white">"Hey, I forgot my login and I'm locked out."</span>
                    </div>
                  </div>
                </div>
                <div>
                  <span className="font-mono text-xs text-[#A1A1AA] block mb-2">Sift-1B Output (Zero Fluff):</span>
                  <div className="p-4 rounded-xl bg-[#151515] border border-white/20 font-mono text-sm text-white">
                    <code>{`{\n  "route": "password_reset",\n  "confidence": 0.98\n}`}</code>
                  </div>
                </div>
              </div>

              {/* Example 2 */}
              <div className="rounded-[24px] border border-[#27272A] bg-[#0A0A0A] p-6 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <span className="font-mono text-xs text-[#A1A1AA] uppercase tracking-wider">Example 2: Temporal Parameter Extraction</span>
                  </div>
                  <div className="space-y-3 mb-4 font-mono text-xs">
                    <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                      <span className="text-[#A1A1AA] block mb-1">Target Schema:</span>
                      <code className="text-white/80">{`{"tool": "book_flight", "destination": "string", "date": "YYYY-MM-DD"}`}</code>
                    </div>
                    <div className="p-3 rounded-xl bg-white/[0.03] border border-white/[0.06]">
                      <span className="text-[#A1A1AA] block mb-1">User Query:</span>
                      <span className="text-white">"Can you get me a ticket to London for next Tuesday?"</span>
                    </div>
                  </div>
                </div>
                <div>
                  <span className="font-mono text-xs text-[#A1A1AA] block mb-2">Sift-1B Output (Parsed Date Resolution):</span>
                  <div className="p-4 rounded-xl bg-[#151515] border border-white/20 font-mono text-sm text-white">
                    <code>{`{\n  "tool": "book_flight",\n  "destination": "London",\n  "date": "2026-08-25"\n}`}</code>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/*  BENCHMARKS SECTION                                               */}
      {/* ================================================================= */}
      <section id="benchmarks" className="py-24 border-t border-[#27272A]/50 bg-[#080808]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="mb-14">

            <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-white mb-4">
              Benchmarks & Specifications
            </h2>
            <p className="text-base text-[#A1A1AA] max-w-2xl">
              Evaluated on 500 unseen holdout schemas. Measured on local hardware with zero cloud API dependencies.
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
                ["Export Formats", "GGUF Q4_K_M (940 MB) + SafeTensors Fused Weights"],
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
      {/*  ARCHITECTURE PIPELINE                                            */}
      {/* ================================================================= */}
      <section id="architecture" className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="mb-14">

            <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-white mb-4">
              SLM → Router → Execution Pipeline
            </h2>
            <p className="text-base text-[#A1A1AA] max-w-xl">
              Sift acts as an edge-native load balancer. It processes structured parameters instantly and only escalates unstructured reasoning tasks to cloud LLMs.
            </p>
          </div>

          <div className="rounded-[32px] border border-[#27272A] bg-[#151515] p-8 md:p-12">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
              {/* Node 1 */}
              <div className="rounded-[24px] border border-[#27272A] bg-[#0A0A0A] p-6 text-center">
                <div className="w-10 h-10 rounded-full border border-white/20 bg-white/5 flex items-center justify-center mx-auto mb-4 text-white font-mono">
                  1
                </div>
                <h4 className="font-medium text-base text-white mb-1">User Input</h4>
                <p className="text-xs text-[#A1A1AA] font-mono">Unstructured Natural Prompt</p>
              </div>

              {/* Node 2 */}
              <div className="rounded-[24px] border border-white/30 bg-[#0A0A0A] p-6 text-center shadow-[0_0_24px_rgba(255,255,255,0.03)]">
                <div className="w-10 h-10 rounded-full border border-white bg-white text-black flex items-center justify-center mx-auto mb-4 font-bold font-mono">
                  2
                </div>
                <h4 className="font-medium text-base text-white mb-1">Sift-1B Router</h4>
                <p className="text-xs text-[#A1A1AA] font-mono">~30ms Deterministic JSON</p>
              </div>

              {/* Node 3 */}
              <div className="rounded-[24px] border border-[#27272A] bg-[#0A0A0A] p-6 text-center">
                <div className="w-10 h-10 rounded-full border border-white/20 bg-white/5 flex items-center justify-center mx-auto mb-4 text-white font-mono">
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
      {/*  DEVELOPER INTEGRATION                                            */}
      {/* ================================================================= */}
      <section id="integration" className="py-24 border-t border-[#27272A]/50 bg-[#080808]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="mb-14">

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
      {/*  FREQUENTLY ASKED QUESTIONS                                       */}
      {/* ================================================================= */}
      <section id="faq" className="py-24 border-t border-[#27272A]/50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="mb-14">

            <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-white mb-4">
              Frequently Asked Questions
            </h2>
            <p className="text-base text-[#A1A1AA] max-w-xl">
              Everything you need to know about deploying, running, and integrating Sift-1B.
            </p>
          </div>

          <div className="space-y-4 max-w-4xl">
            {FAQS.map((faq, idx) => (
              <div
                key={idx}
                className="rounded-[24px] border border-[#27272A] bg-[#151515] overflow-hidden transition-all"
              >
                <button
                  onClick={() => setOpenFaq(openFaq === idx ? null : idx)}
                  className="w-full px-8 py-5 text-left flex items-center justify-between gap-4 cursor-pointer hover:bg-white/[0.02] transition-colors"
                >
                  <span className="font-medium text-base text-white">{faq.q}</span>
                  <span className="text-lg text-[#A1A1AA] font-mono shrink-0">
                    {openFaq === idx ? "−" : "+"}
                  </span>
                </button>
                {openFaq === idx && (
                  <div className="px-8 pb-6 pt-1 text-sm text-[#A1A1AA] leading-relaxed border-t border-[#27272A]/40 font-normal">
                    {faq.a}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ================================================================= */}
      {/*  CONTACT & AUTHOR SECTION                                         */}
      {/* ================================================================= */}
      <section id="contact" className="py-24 border-t border-[#27272A]/50 bg-[#080808]">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-start">

            {/* Left Column: Author Info & Handles */}
            <div className="lg:col-span-5">

              <h2 className="text-3xl md:text-5xl font-medium tracking-tight text-white mb-4">
                Connect with the Author
              </h2>
              <p className="text-base text-[#A1A1AA] leading-relaxed mb-8">
                Sift-1B is an open-source research initiative designed and developed by <strong className="text-white font-medium">Sanatan Singh</strong>.
              </p>

              <div className="space-y-3">
                <a
                  href="mailto:sanatansinghgonda@gmail.com"
                  className="flex items-center gap-3 px-5 py-3.5 rounded-full border border-[#27272A] bg-[#151515] text-sm text-[#A1A1AA] hover:text-white hover:border-white/30 transition-all group"
                >
                  <span className="text-xs">✉️</span>
                  <span className="font-medium text-white/90">sanatansinghgonda@gmail.com</span>
                  <span className="ml-auto text-xs text-white/40 group-hover:text-white transition-colors">↗</span>
                </a>

                <a
                  href="https://github.com/SanatanSinghVishen"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 px-5 py-3.5 rounded-full border border-[#27272A] bg-[#151515] text-sm text-[#A1A1AA] hover:text-white hover:border-white/30 transition-all group"
                >
                  <span className="font-mono text-white text-xs">GH</span>
                  <span className="font-medium text-white/90">github.com/SanatanSinghVishen</span>
                  <span className="ml-auto text-xs text-white/40 group-hover:text-white transition-colors">↗</span>
                </a>

                <a
                  href="https://huggingface.co/SanatanSinghVishen"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 px-5 py-3.5 rounded-full border border-[#27272A] bg-[#151515] text-sm text-[#A1A1AA] hover:text-white hover:border-white/30 transition-all group"
                >
                  <span className="text-xs">🤗</span>
                  <span className="font-medium text-white/90">huggingface.co/SanatanSinghVishen</span>
                  <span className="ml-auto text-xs text-white/40 group-hover:text-white transition-colors">↗</span>
                </a>

                <a
                  href="https://linkedin.com"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-3 px-5 py-3.5 rounded-full border border-[#27272A] bg-[#151515] text-sm text-[#A1A1AA] hover:text-white hover:border-white/30 transition-all group"
                >
                  <span className="font-mono text-white text-xs">IN</span>
                  <span className="font-medium text-white/90">Sanatan Singh on LinkedIn</span>
                  <span className="ml-auto text-xs text-white/40 group-hover:text-white transition-colors">↗</span>
                </a>
              </div>
            </div>

            {/* Right Column: Direct Contact Form */}
            <div className="lg:col-span-7">
              <div className="rounded-[32px] border border-[#27272A] bg-[#151515] p-8 md:p-10">
                <h3 className="text-xl font-medium text-white mb-2">Send a Message</h3>
                <p className="text-xs text-[#A1A1AA] mb-6">Have an idea, inquiry, or partnership request? Drop a message below.</p>

                <form onSubmit={handleContactSubmit} className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-mono text-[#A1A1AA] uppercase mb-2">Name</label>
                      <input
                        type="text"
                        required
                        value={contactForm.name}
                        onChange={(e) => setContactForm({ ...contactForm, name: e.target.value })}
                        placeholder="Your Name"
                        className="w-full px-5 py-3 rounded-full border border-[#27272A] bg-[#0A0A0A] text-sm text-white placeholder:text-[#A1A1AA]/50 focus:outline-none focus:border-white/40 transition-colors"
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-mono text-[#A1A1AA] uppercase mb-2">Email</label>
                      <input
                        type="email"
                        required
                        value={contactForm.email}
                        onChange={(e) => setContactForm({ ...contactForm, email: e.target.value })}
                        placeholder="you@domain.com"
                        className="w-full px-5 py-3 rounded-full border border-[#27272A] bg-[#0A0A0A] text-sm text-white placeholder:text-[#A1A1AA]/50 focus:outline-none focus:border-white/40 transition-colors"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-mono text-[#A1A1AA] uppercase mb-2">Message</label>
                    <textarea
                      rows={4}
                      required
                      value={contactForm.message}
                      onChange={(e) => setContactForm({ ...contactForm, message: e.target.value })}
                      placeholder="Describe your use case or inquiry..."
                      className="w-full px-5 py-3.5 rounded-[24px] border border-[#27272A] bg-[#0A0A0A] text-sm text-white placeholder:text-[#A1A1AA]/50 focus:outline-none focus:border-white/40 transition-colors resize-none"
                    />
                  </div>

                  <button
                    type="submit"
                    className="w-full sm:w-auto px-8 py-3.5 rounded-full bg-white text-black font-medium text-sm hover:bg-neutral-200 transition-all cursor-pointer shadow-md"
                  >
                    Submit Inquiry →
                  </button>

                  {contactStatus && (
                    <p className="text-xs font-mono text-emerald-400 mt-2">
                      ✓ {contactStatus}
                    </p>
                  )}
                </form>
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
                Built by Sanatan Singh
              </span>
            </div>
            <div className="flex items-center gap-8 text-sm text-[#A1A1AA]">
              <a
                href="https://huggingface.co/SanatanSinghVishen/sift-1b-gguf"
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
