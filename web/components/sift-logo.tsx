"use client";

export function SiftLogo({ className = "w-16 h-16" }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 240 228"
      className={className}
      fill="none"
    >
      <defs>
        <filter
          id="material-shadow"
          x="-20%"
          y="-20%"
          width="140%"
          height="140%"
        >
          <feDropShadow
            dx="0"
            dy="8"
            stdDeviation="12"
            floodColor="#0f172a"
            floodOpacity="0.15"
          />
        </filter>
        <linearGradient id="left-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#4338ca" />
          <stop offset="100%" stopColor="#312E81" />
        </linearGradient>
        <linearGradient id="right-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#fb7185" />
          <stop offset="100%" stopColor="#e11d48" />
        </linearGradient>
      </defs>

      {/* Left Half (Deep Indigo) — represents Extraction */}
      <path
        d="M 114 28 L 46 96 Q 40 102 46 108 L 114 176 Z"
        fill="url(#left-grad)"
        filter="url(#material-shadow)"
      />

      {/* Right Half (Soft Coral) — represents Routing */}
      <path
        d="M 126 52 L 194 120 Q 200 126 194 132 L 126 200 Z"
        fill="url(#right-grad)"
        filter="url(#material-shadow)"
      />
    </svg>
  );
}
