import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Sift-1B — Deterministic Function Calling SLM",
  description:
    "A 1.5B-parameter Small Language Model fine-tuned via QLoRA + DPO for deterministic JSON function calling. Runs entirely on your hardware.",
  keywords: [
    "SLM",
    "function calling",
    "JSON",
    "QLoRA",
    "DPO",
    "fine-tuning",
    "edge AI",
    "Qwen",
    "open source",
  ],
  authors: [{ name: "Sanatan Singh Vishen" }],
  openGraph: {
    title: "Sift-1B — Extract the signal. Route the action.",
    description:
      "Ultra-fast 1.5B SLM for deterministic function calling. Zero API costs. Total data privacy.",
    type: "website",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} dark h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        {children}
      </body>
    </html>
  );
}
