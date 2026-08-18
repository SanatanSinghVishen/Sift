"use client";

interface BenchmarkCardProps {
  label: string;
  value: string;
  target?: string;
  icon: string;
  color?: "indigo" | "rose" | "emerald" | "amber";
}

export function BenchmarkCard({
  label,
  value,
  target,
  icon,
}: BenchmarkCardProps) {
  return (
    <div
      className="relative overflow-hidden rounded-[32px] border border-[#27272A] bg-[#151515] p-6 transition-all duration-300 hover:border-white/20 hover:-translate-y-1 group"
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-[#A1A1AA] font-mono mb-1.5 uppercase tracking-wider">
            {label}
          </p>
          <p className="text-3xl font-medium tracking-tight text-[#FFFFFF]">
            {value}
          </p>
          {target && (
            <p className="text-xs text-[#A1A1AA]/80 font-mono mt-2">
              {target}
            </p>
          )}
        </div>
        <span className="text-2xl opacity-80 group-hover:scale-110 transition-transform">{icon}</span>
      </div>

      {/* Subtle top sheen */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />
    </div>
  );
}
