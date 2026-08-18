"use client";

import React, { useState, useEffect, useRef } from "react";

export function Hero3DStack() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [rotate, setRotate] = useState({ x: 54, y: 0, z: -32 });
  const [targetRotate, setTargetRotate] = useState({ x: 54, y: 0, z: -32 });

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const { innerWidth, innerHeight } = window;
      // Normalized coordinates from -1 to 1
      const nx = (e.clientX / innerWidth) * 2 - 1;
      const ny = (e.clientY / innerHeight) * 2 - 1;

      // Base isometric tilt + cursor interaction
      const baseTiltX = 54;
      const baseTiltZ = -32;

      setTargetRotate({
        x: baseTiltX - ny * 12,
        y: nx * 14,
        z: baseTiltZ + nx * 8,
      });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  // Smooth lerp damping loop
  useEffect(() => {
    let animFrame: number;
    const lerp = (start: number, end: number, factor: number) =>
      start + (end - start) * factor;

    const loop = () => {
      setRotate((prev) => ({
        x: lerp(prev.x, targetRotate.x, 0.08),
        y: lerp(prev.y, targetRotate.y, 0.08),
        z: lerp(prev.z, targetRotate.z, 0.08),
      }));
      animFrame = requestAnimationFrame(loop);
    };

    animFrame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animFrame);
  }, [targetRotate]);

  // 6 stacked layers with increasing height/depth
  const layers = [
    { id: 0, z: 0, opacity: 0.25, blur: "blur-[1px]", label: "Inference Engine" },
    { id: 1, z: 28, opacity: 0.35, blur: "blur-[0.5px]", label: "LoRA Adapters" },
    { id: 2, z: 56, opacity: 0.5, blur: "none", label: "DPO Alignment" },
    { id: 3, z: 84, opacity: 0.65, blur: "none", label: "SFT Weights" },
    { id: 4, z: 112, opacity: 0.85, blur: "none", label: "Qwen-1.5B Base" },
    { id: 5, z: 144, opacity: 1.0, blur: "none", label: "Sift Core Routing", isTop: true },
  ];

  return (
    <div
      ref={containerRef}
      className="relative w-full max-w-[540px] h-[480px] md:h-[540px] flex items-center justify-center perspective-container select-none"
    >
      {/* Ambient background glow behind the stack */}
      <div className="absolute w-[360px] h-[360px] rounded-full bg-white/[0.03] blur-[90px] pointer-events-none" />

      {/* 3D Rotational Stage */}
      <div
        className="relative w-[320px] h-[320px] sm:w-[360px] sm:h-[360px] preserve-3d transition-transform duration-75 ease-out"
        style={{
          transform: `rotateX(${rotate.x}deg) rotateY(${rotate.y}deg) rotateZ(${rotate.z}deg)`,
        }}
      >
        {layers.map((layer) => (
          <div
            key={layer.id}
            className="absolute inset-0 rounded-[32px] preserve-3d transition-all duration-300 pointer-events-none"
            style={{
              transform: `translateZ(${layer.z}px)`,
            }}
          >
            {/* Main Glass/Matte Plate Body */}
            <div
              className={`w-full h-full rounded-[32px] border border-white/[0.12] plate-edge relative overflow-hidden transition-all duration-500`}
              style={{
                backgroundColor: layer.isTop ? "#151515" : "#0d0d0d",
                opacity: layer.opacity,
              }}
            >
              {/* Subtle top surface light gradient */}
              <div className="absolute inset-0 plate-highlight opacity-80" />

              {/* 4 Corner Precision Pins / Hardware Dots */}
              <div className="absolute top-4 left-4 w-2 h-2 rounded-full border border-white/20 bg-white/10 shadow-[0_0_8px_rgba(255,255,255,0.2)]" />
              <div className="absolute top-4 right-4 w-2 h-2 rounded-full border border-white/20 bg-white/10 shadow-[0_0_8px_rgba(255,255,255,0.2)]" />
              <div className="absolute bottom-4 left-4 w-2 h-2 rounded-full border border-white/20 bg-white/10 shadow-[0_0_8px_rgba(255,255,255,0.2)]" />
              <div className="absolute bottom-4 right-4 w-2 h-2 rounded-full border border-white/20 bg-white/10 shadow-[0_0_8px_rgba(255,255,255,0.2)]" />

              {/* Subtle Grid / Texture for Top Layer */}
              {layer.isTop && (
                <div
                  className="absolute inset-0 opacity-[0.08]"
                  style={{
                    backgroundImage: `radial-gradient(rgba(255,255,255,0.3) 1px, transparent 1px)`,
                    backgroundSize: "16px 16px",
                  }}
                />
              )}

              {/* Top Layer Sift Brand Monogram */}
              {layer.isTop && (
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <div className="relative flex items-center justify-center">
                    {/* Glowing S monogram */}
                    <svg
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="w-16 h-16 text-white/90 drop-shadow-[0_0_16px_rgba(255,255,255,0.4)]"
                    >
                      <path d="M4 8a4 4 0 0 1 4-4h8a4 4 0 0 1 4 4v1a3 3 0 0 1-3 3H7a3 3 0 0 0-3 3v1a4 4 0 0 0 4 4h8a4 4 0 0 0 4-4" />
                    </svg>
                  </div>
                  <span className="font-mono text-[11px] uppercase tracking-[0.25em] text-white/50 mt-3 font-medium">
                    Sift · 1.5B
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
