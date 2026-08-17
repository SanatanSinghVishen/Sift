"use client";

import { useState } from "react";

interface BenchmarkCardProps {
  label: string;
  value: string;
  target?: string;
  icon: string;
  color: "indigo" | "rose" | "emerald" | "amber";
}

const colorMap = {
  indigo: {
    bg: "from-indigo-500/10 to-indigo-500/5",
    border: "border-indigo-500/20",
    text: "text-indigo-400",
    glow: "shadow-indigo-500/5",
  },
  rose: {
    bg: "from-rose-500/10 to-rose-500/5",
    border: "border-rose-500/20",
    text: "text-rose-400",
    glow: "shadow-rose-500/5",
  },
  emerald: {
    bg: "from-emerald-500/10 to-emerald-500/5",
    border: "border-emerald-500/20",
    text: "text-emerald-400",
    glow: "shadow-emerald-500/5",
  },
  amber: {
    bg: "from-amber-500/10 to-amber-500/5",
    border: "border-amber-500/20",
    text: "text-amber-400",
    glow: "shadow-amber-500/5",
  },
};

export function BenchmarkCard({
  label,
  value,
  target,
  icon,
  color,
}: BenchmarkCardProps) {
  const c = colorMap[color];

  return (
    <div
      className={`
        relative overflow-hidden rounded-2xl border ${c.border}
        bg-gradient-to-br ${c.bg} backdrop-blur-sm
        p-6 transition-all duration-300 hover:scale-[1.02]
        hover:shadow-lg ${c.glow}
      `}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-muted-foreground font-medium mb-1">
            {label}
          </p>
          <p className={`text-3xl font-bold tracking-tight ${c.text}`}>
            {value}
          </p>
          {target && (
            <p className="text-xs text-muted-foreground mt-1.5">
              Target: {target}
            </p>
          )}
        </div>
        <span className="text-2xl">{icon}</span>
      </div>

      {/* Subtle glow effect */}
      <div
        className={`absolute -bottom-8 -right-8 w-24 h-24 rounded-full bg-gradient-to-br ${c.bg} opacity-40 blur-2xl`}
      />
    </div>
  );
}
