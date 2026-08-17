"use client";

import { useState } from "react";
import { SiftLogo } from "@/components/sift-logo";
import { CodeBlock } from "@/components/code-block";
import { BenchmarkCard } from "@/components/benchmark-card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";

// =============================================================================
// Code Snippets
// =============================================================================

const CODE_SNIPPETS = {
  python: `from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="sift-1b",
    messages=[
        {"role": "system", "content": "You are a function calling agent."},
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

  docker: `# Pull and run the inference server
docker run -d -p 8000:8000 --gpus all \\
  ghcr.io/sanatansinghvishen/sift-1b:latest

# Or build from source
git clone https://github.com/SanatanSinghVishen/Sift.git
cd Sift
docker build -t sift-1b .
docker run -d -p 8000:8000 sift-1b`,
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
    <main className="flex flex-col min-h-screen">
      {/* ================================================================= */}
      {/*  NAVBAR                                                           */}
      {/* ================================================================= */}
      <nav className="sticky top-0 z-50 border-b border-white/[0.06] bg-background/80 backdrop-blur-xl">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <SiftLogo className="w-8 h-8" />
            <span className="font-semibold text-lg tracking-tight">Sift</span>
            <Badge variant="secondary" className="text-[10px] font-mono">
              v1.0
            </Badge>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm text-muted-foreground">
            <a href="#benchmarks" className="hover:text-foreground transition-colors">
              Benchmarks
            </a>
            <a href="#integration" className="hover:text-foreground transition-colors">
              Integration
            </a>
            <a href="#architecture" className="hover:text-foreground transition-colors">
              Architecture
            </a>
            <a
              href="https://huggingface.co/SanatanSinghVishen/sift-1b"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-foreground transition-colors"
            >
              🤗 Model
            </a>
            <a
              href="https://github.com/SanatanSinghVishen/Sift"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-foreground transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </nav>

      {/* ================================================================= */}
      {/*  HERO SECTION                                                     */}
      {/* ================================================================= */}
      <section className="relative overflow-hidden">
        {/* Background gradient orbs */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-[-20%] left-[-10%] w-[600px] h-[600px] rounded-full bg-indigo-600/[0.07] blur-[120px]" />
          <div className="absolute bottom-[-20%] right-[-10%] w-[500px] h-[500px] rounded-full bg-rose-600/[0.07] blur-[120px]" />
          <div className="absolute top-[40%] left-[50%] w-[300px] h-[300px] rounded-full bg-violet-600/[0.05] blur-[100px]" />
        </div>

        {/* Grid pattern overlay */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
                              linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)`,
            backgroundSize: "64px 64px",
          }}
        />

        <div className="relative max-w-6xl mx-auto px-6 pt-24 pb-20">
          <div className="flex flex-col items-center text-center">
            {/* Logo */}
            <div className="mb-8 animate-float">
              <SiftLogo className="w-24 h-24 md:w-28 md:h-28" />
            </div>

            {/* Heading */}
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-4">
              <span className="bg-gradient-to-r from-slate-100 via-slate-200 to-slate-300 bg-clip-text text-transparent">
                Sift-1B
              </span>
            </h1>

            {/* Tagline */}
            <p className="text-xl md:text-2xl text-muted-foreground font-light mb-3 tracking-wide">
              Extract the signal. Route the action.
            </p>

            {/* Description */}
            <p className="max-w-2xl text-muted-foreground/80 text-base md:text-lg leading-relaxed mb-10">
              A 1.5B-parameter SLM fine-tuned via{" "}
              <span className="text-indigo-400 font-medium">QLoRA</span> +{" "}
              <span className="text-rose-400 font-medium">DPO</span> for
              deterministic function calling. Zero conversational fluff. Strict
              JSON. Runs entirely on your hardware.
            </p>

            {/* CLI Commands */}
            <div className="w-full max-w-xl space-y-3 mb-10">
              {CLI_COMMANDS.map(({ label, cmd }) => (
                <div
                  key={label}
                  className="flex items-center gap-3 rounded-xl border border-white/[0.08] bg-white/[0.02] px-4 py-3 backdrop-blur-sm group hover:border-white/[0.15] transition-colors"
                >
                  <span className="text-xs text-muted-foreground font-mono w-14 shrink-0">
                    {label}
                  </span>
                  <code className="flex-1 text-sm font-mono text-slate-300 truncate">
                    $ {cmd}
                  </code>
                  <button
                    onClick={() => copyCommand(cmd)}
                    className="text-xs px-3 py-1 rounded-md bg-white/[0.06] hover:bg-white/[0.12] text-muted-foreground hover:text-foreground transition-all font-mono shrink-0 cursor-pointer"
                  >
                    {copiedCmd === cmd ? "✓" : "Copy"}
                  </button>
                </div>
              ))}
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-wrap items-center justify-center gap-4">
              <a
                href="#integration"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-600 to-indigo-500 text-white font-medium text-sm hover:from-indigo-500 hover:to-indigo-400 transition-all shadow-lg shadow-indigo-500/20 hover:shadow-indigo-500/30"
              >
                Get Started
              </a>
              <a
                href="https://huggingface.co/SanatanSinghVishen/sift-1b"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl border border-white/[0.1] bg-white/[0.03] text-sm font-medium hover:bg-white/[0.06] transition-all"
              >
                🤗 View on Hugging Face
              </a>
              <a
                href="https://github.com/SanatanSinghVishen/Sift"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl border border-white/[0.1] bg-white/[0.03] text-sm font-medium hover:bg-white/[0.06] transition-all"
              >
                ⭐ GitHub
              </a>
            </div>
          </div>
        </div>
      </section>

      <Separator className="opacity-10" />

      {/* ================================================================= */}
      {/*  BENCHMARKS                                                       */}
      {/* ================================================================= */}
      <section id="benchmarks" className="py-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-12">
            <Badge variant="secondary" className="mb-4 font-mono text-xs">
              Performance
            </Badge>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-3">
              Benchmarks & Specifications
            </h2>
            <p className="text-muted-foreground max-w-lg mx-auto">
              Evaluated on 500 unseen schemas not present in the training dataset.
              All metrics measured on an NVIDIA RTX 3050 (4 GB VRAM).
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <BenchmarkCard
              label="Schema Adherence"
              value="98.4%"
              target="≥ 95%"
              icon="🎯"
              color="indigo"
            />
            <BenchmarkCard
              label="Hallucination Rate"
              value="<2%"
              target="≤ 2%"
              icon="🛡️"
              color="emerald"
            />
            <BenchmarkCard
              label="TTFT (Q4_K_M)"
              value="~32ms"
              icon="⚡"
              color="amber"
            />
            <BenchmarkCard
              label="VRAM Usage"
              value="~1.2 GB"
              icon="💾"
              color="rose"
            />
          </div>

          {/* Training specs table */}
          <div className="mt-12 rounded-2xl border border-white/[0.08] bg-white/[0.02] overflow-hidden">
            <div className="px-6 py-4 border-b border-white/[0.06]">
              <h3 className="font-semibold text-sm">Training Configuration</h3>
            </div>
            <div className="divide-y divide-white/[0.04]">
              {[
                ["Base Model", "Qwen2.5-1.5B-Instruct"],
                ["Quantization", "QLoRA (4-bit NF4)"],
                ["LoRA Rank / Alpha", "r=16, α=32"],
                ["SFT Dataset", "Salesforce/xlam-function-calling-60k (10k sample)"],
                ["DPO Dataset", "Synthetically mutated preference pairs (5 strategies)"],
                ["Alignment", "SFT → DPO (β=0.1, sigmoid loss)"],
                ["Training Hardware", "NVIDIA RTX 3050 4 GB · Lenovo IdeaPad Gaming 3"],
                ["Framework", "Unsloth + TRL + PEFT"],
                ["Export Format", "GGUF Q4_K_M + SafeTensors LoRA"],
              ].map(([key, val]) => (
                <div
                  key={key}
                  className="flex items-center px-6 py-3 text-sm hover:bg-white/[0.02] transition-colors"
                >
                  <span className="w-48 text-muted-foreground shrink-0 font-medium">
                    {key}
                  </span>
                  <span className="font-mono text-slate-300">{val}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <Separator className="opacity-10" />

      {/* ================================================================= */}
      {/*  WHY SIFT                                                         */}
      {/* ================================================================= */}
      <section className="py-20 relative">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-[50%] left-[20%] w-[400px] h-[400px] rounded-full bg-indigo-600/[0.04] blur-[100px]" />
        </div>
        <div className="relative max-w-6xl mx-auto px-6">
          <div className="text-center mb-12">
            <Badge variant="secondary" className="mb-4 font-mono text-xs">
              Why Sift?
            </Badge>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-3">
              The problem with cloud LLMs
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                icon: "💸",
                title: "Zero API Costs",
                desc: "Stop paying per-token for simple routing tasks. Run 50,000+ function calls daily for $0 on your own hardware.",
              },
              {
                icon: "🔒",
                title: "Total Data Privacy",
                desc: "Sensitive data never leaves your infrastructure. No third-party API calls. GDPR, HIPAA, and SOC 2 compliant by architecture.",
              },
              {
                icon: "🎯",
                title: "Zero-Retry Reliability",
                desc: "DPO-aligned to eliminate markdown wrappers, conversational preamble, and hallucinated parameters. Clean JSON, first try.",
              },
            ].map((item) => (
              <div
                key={item.title}
                className="rounded-2xl border border-white/[0.08] bg-white/[0.02] p-6 hover:border-white/[0.15] transition-all duration-300"
              >
                <span className="text-3xl mb-4 block">{item.icon}</span>
                <h3 className="font-semibold text-lg mb-2">{item.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {item.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <Separator className="opacity-10" />

      {/* ================================================================= */}
      {/*  INTEGRATION CODE SNIPPETS                                        */}
      {/* ================================================================= */}
      <section id="integration" className="py-20">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-12">
            <Badge variant="secondary" className="mb-4 font-mono text-xs">
              Integration
            </Badge>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-3">
              Drop-in replacement for cloud APIs
            </h2>
            <p className="text-muted-foreground max-w-lg mx-auto">
              Point your existing OpenAI SDK to <code className="font-mono text-xs bg-white/[0.06] px-1.5 py-0.5 rounded">localhost:8000</code> and
              start making function calls instantly.
            </p>
          </div>

          <Tabs defaultValue="python" className="w-full">
            <TabsList className="w-full max-w-md mx-auto grid grid-cols-4 mb-6">
              <TabsTrigger value="python" className="font-mono text-xs">
                Python
              </TabsTrigger>
              <TabsTrigger value="typescript" className="font-mono text-xs">
                TypeScript
              </TabsTrigger>
              <TabsTrigger value="curl" className="font-mono text-xs">
                cURL
              </TabsTrigger>
              <TabsTrigger value="docker" className="font-mono text-xs">
                Docker
              </TabsTrigger>
            </TabsList>

            <TabsContent value="python">
              <CodeBlock
                code={CODE_SNIPPETS.python}
                language="python"
                filename="main.py"
              />
            </TabsContent>
            <TabsContent value="typescript">
              <CodeBlock
                code={CODE_SNIPPETS.typescript}
                language="typescript"
                filename="index.ts"
              />
            </TabsContent>
            <TabsContent value="curl">
              <CodeBlock
                code={CODE_SNIPPETS.curl}
                language="bash"
                filename="terminal"
              />
            </TabsContent>
            <TabsContent value="docker">
              <CodeBlock
                code={CODE_SNIPPETS.docker}
                language="bash"
                filename="terminal"
              />
            </TabsContent>
          </Tabs>
        </div>
      </section>

      <Separator className="opacity-10" />

      {/* ================================================================= */}
      {/*  ARCHITECTURE                                                     */}
      {/* ================================================================= */}
      <section id="architecture" className="py-20 relative">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute bottom-[30%] right-[10%] w-[400px] h-[400px] rounded-full bg-rose-600/[0.04] blur-[100px]" />
        </div>
        <div className="relative max-w-6xl mx-auto px-6">
          <div className="text-center mb-12">
            <Badge variant="secondary" className="mb-4 font-mono text-xs">
              Architecture
            </Badge>
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-3">
              SLM → Router → LLM Pipeline
            </h2>
            <p className="text-muted-foreground max-w-xl mx-auto">
              Sift acts as a high-speed AI load balancer. It handles cheap,
              structured routing tasks instantly — and only escalates complex
              reasoning to expensive frontier models.
            </p>
          </div>

          {/* Architecture diagram */}
          <div className="rounded-2xl border border-white/[0.08] bg-white/[0.02] p-8 md:p-12">
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4 items-center">
              {/* Step 1 */}
              <div className="text-center p-4 rounded-xl border border-indigo-500/20 bg-indigo-500/[0.05]">
                <div className="text-2xl mb-2">📨</div>
                <p className="text-xs font-semibold text-indigo-400">Input</p>
                <p className="text-[11px] text-muted-foreground mt-1">
                  User prompt
                </p>
              </div>

              {/* Arrow */}
              <div className="hidden md:flex justify-center text-muted-foreground/40">
                →
              </div>

              {/* Step 2 */}
              <div className="text-center p-4 rounded-xl border border-rose-500/20 bg-rose-500/[0.05]">
                <div className="text-2xl mb-2">⚡</div>
                <p className="text-xs font-semibold text-rose-400">
                  Sift-1B (SLM)
                </p>
                <p className="text-[11px] text-muted-foreground mt-1">
                  ~32ms · Strict JSON
                </p>
              </div>

              {/* Arrow */}
              <div className="hidden md:flex justify-center text-muted-foreground/40">
                →
              </div>

              {/* Step 3 */}
              <div className="text-center p-4 rounded-xl border border-emerald-500/20 bg-emerald-500/[0.05]">
                <div className="text-2xl mb-2">🔧</div>
                <p className="text-xs font-semibold text-emerald-400">
                  Tool Execution
                </p>
                <p className="text-[11px] text-muted-foreground mt-1">
                  API / DB / Backend
                </p>
              </div>
            </div>

            {/* Fallback path */}
            <div className="mt-8 pt-6 border-t border-white/[0.06]">
              <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground">
                <span className="px-3 py-1.5 rounded-lg border border-amber-500/20 bg-amber-500/[0.05] text-amber-400">
                  Validation fails?
                </span>
                <span>→</span>
                <span className="px-3 py-1.5 rounded-lg border border-violet-500/20 bg-violet-500/[0.05] text-violet-400">
                  Escalate to Frontier LLM
                </span>
                <span>→</span>
                <span className="px-3 py-1.5 rounded-lg border border-emerald-500/20 bg-emerald-500/[0.05] text-emerald-400">
                  Deep reasoning + synthesis
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      <Separator className="opacity-10" />

      {/* ================================================================= */}
      {/*  FOOTER                                                           */}
      {/* ================================================================= */}
      <footer className="py-12 border-t border-white/[0.06]">
        <div className="max-w-6xl mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <SiftLogo className="w-6 h-6" />
              <span className="text-sm font-medium">Sift-1B</span>
              <span className="text-xs text-muted-foreground">
                Built by Sanatan Singh Vishen
              </span>
            </div>
            <div className="flex items-center gap-6 text-sm text-muted-foreground">
              <a
                href="https://huggingface.co/SanatanSinghVishen/sift-1b"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-foreground transition-colors"
              >
                Hugging Face
              </a>
              <a
                href="https://github.com/SanatanSinghVishen/Sift"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-foreground transition-colors"
              >
                GitHub
              </a>
              <span>MIT License</span>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}
