import React, { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const TechDetails = () => {
    const containerRef = useRef(null);

    useGSAP(() => {
        const tl = gsap.timeline({
            scrollTrigger: {
                trigger: containerRef.current,
                start: 'top 80%',
                end: 'bottom 20%',
                toggleActions: 'play none none reverse',
            }
        });

        tl.from('.tech-item', {
            x: -50,
            opacity: 0,
            duration: 0.8,
            stagger: 0.1,
            ease: 'power3.out',
        })
            .from('.tech-visual', {
                x: 50,
                opacity: 0,
                duration: 1,
                ease: 'power3.out',
            }, '-=0.6');

    }, { scope: containerRef });

    return (
        <div id="tech-details" ref={containerRef} className="py-24 relative overflow-hidden">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">

                    {/* Text Content */}
                    <div>
                        <h2 className="text-[#0ae448] font-mono text-sm tracking-wider mb-4 uppercase">Architecture</h2>
                        <h3 className="text-4xl font-bold text-white mb-6">Cross-Scale Residual Network</h3>
                        <p className="text-gray-400 text-lg mb-8 leading-relaxed">
                            The CSRN model utilizes a dual-branch architecture to extract high-frequency spatial details from 30m optical data and fuse them with 100m thermal data.
                        </p>

                        <ul className="space-y-6">
                            {[
                                { title: 'Deep Feature Extraction', desc: 'ResNet-50 backbone for optical feature mining.' },
                                { title: 'Attention Mechanism', desc: 'Spatial and channel attention for adaptive fusion.' },
                                { title: 'PixelShuffle Upsampling', desc: 'Sub-pixel convolution for artifact-free scaling.' }
                            ].map((item, i) => (
                                <li key={i} className="tech-item flex gap-4">
                                    <div className="flex-none w-12 h-12 rounded-full bg-white/5 border border-white/10 flex items-center justify-center text-[#9d95ff] font-bold">
                                        {i + 1}
                                    </div>
                                    <div>
                                        <h4 className="text-white font-semibold text-lg">{item.title}</h4>
                                        <p className="text-gray-500">{item.desc}</p>
                                    </div>
                                </li>
                            ))}
                        </ul>
                    </div>

                    {/* Visual/Diagram */}
                    <div className="tech-visual relative">
                        <div className="absolute inset-0 bg-gradient-to-tr from-[#0ae448]/20 to-[#9d95ff]/20 blur-3xl rounded-full opacity-30"></div>
                        <div className="relative bg-white/5 border border-white/10 rounded-2xl p-8 shadow-2xl backdrop-blur-sm flex items-center justify-center">
                            {/* Custom Architecture SVG */}
                            <svg viewBox="0 0 800 400" className="w-full h-auto text-white" fill="none" stroke="currentColor" strokeWidth="2">
                                {/* Input Nodes */}
                                <g transform="translate(50, 100)">
                                    <rect x="0" y="0" width="100" height="80" rx="8" className="fill-white/10 stroke-[#0ae448]" />
                                    <text x="50" y="45" textAnchor="middle" className="fill-white text-xs font-mono" stroke="none">Optical (30m)</text>
                                </g>
                                <g transform="translate(50, 220)">
                                    <rect x="0" y="0" width="100" height="80" rx="8" className="fill-white/10 stroke-[#9d95ff]" />
                                    <text x="50" y="45" textAnchor="middle" className="fill-white text-xs font-mono" stroke="none">Thermal (100m)</text>
                                </g>

                                {/* Feature Extraction */}
                                <g transform="translate(250, 160)">
                                    <rect x="0" y="0" width="120" height="80" rx="8" className="fill-white/5 stroke-white/30" />
                                    <text x="60" y="45" textAnchor="middle" className="fill-white text-xs font-bold" stroke="none">Feature Extraction</text>
                                </g>

                                {/* Fusion Block */}
                                <g transform="translate(450, 160)">
                                    <rect x="0" y="0" width="120" height="80" rx="8" className="fill-white/5 stroke-[#0ae448]" />
                                    <text x="60" y="45" textAnchor="middle" className="fill-white text-xs font-bold" stroke="none">Cross-Scale Fusion</text>
                                </g>

                                {/* Output */}
                                <g transform="translate(650, 160)">
                                    <rect x="0" y="0" width="100" height="80" rx="8" className="fill-[#0ae448]/20 stroke-[#0ae448]" />
                                    <text x="50" y="45" textAnchor="middle" className="fill-white text-xs font-bold" stroke="none">Super Res (30m)</text>
                                </g>

                                {/* Connections */}
                                <path d="M150 140 L250 200" className="stroke-white/20" markerEnd="url(#arrow)" />
                                <path d="M150 260 L250 200" className="stroke-white/20" markerEnd="url(#arrow)" />
                                <path d="M370 200 L450 200" className="stroke-white/20" markerEnd="url(#arrow)" />
                                <path d="M570 200 L650 200" className="stroke-[#0ae448]" markerEnd="url(#arrow-green)" />

                                <defs>
                                    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                                        <path d="M0,0 L0,6 L9,3 z" className="fill-white/20" stroke="none" />
                                    </marker>
                                    <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                                        <path d="M0,0 L0,6 L9,3 z" className="fill-[#0ae448]" stroke="none" />
                                    </marker>
                                </defs>
                            </svg>
                        </div>
                    </div>

                </div>
            </div>
        </div>
    );
};

export default TechDetails;
