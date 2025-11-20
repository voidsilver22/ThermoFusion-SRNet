import React, { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const features = [
    {
        name: 'Super Resolution',
        description: 'Upscales 100m thermal bands to 30m resolution, matching optical bands perfectly.',
        icon: (
            <svg width="24" height="24" className="h-6 w-6 text-sky-400" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9M3.75 20.25v-4.5m0 4.5h4.5m-4.5 0L9 15M20.25 3.75h-4.5m4.5 0v4.5m0-4.5L15 9m5.25 11.25h-4.5m4.5 0v-4.5m0 4.5L15 15" />
            </svg>
        ),
    },
    {
        name: 'Multi-Modal Fusion',
        description: 'Integrates Optical (OLI) and Thermal (TIRS) data for enhanced detail recovery.',
        icon: (
            <svg width="24" height="24" className="h-6 w-6 text-sky-400" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15a4.5 4.5 0 004.5 4.5H18a3.75 3.75 0 001.332-7.257 3 3 0 00-3.758-3.848 5.25 5.25 0 00-10.233 2.33A4.502 4.502 0 002.25 15z" />
            </svg>
        ),
    },
    {
        name: 'Physics-Aware',
        description: 'Maintains physical consistency of Land Surface Temperature (LST) values.',
        icon: (
            <svg width="24" height="24" className="h-6 w-6 text-sky-400" fill="none" viewBox="0 0 24 24" strokeWidth="1.5" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 18v-5.25m0 0a6.01 6.01 0 001.5-.189m-1.5.189a6.01 6.01 0 01-1.5-.189m3.75 7.478a12.06 12.06 0 01-4.5 0m3.75 2.383a14.406 14.406 0 01-3 0M14.25 18v-.192c0-.983.658-1.823 1.508-2.316a7.5 7.5 0 10-7.517 0c.85.493 1.509 1.333 1.509 2.316V18" />
            </svg>
        ),
    },
];

const Features = () => {
    const containerRef = useRef(null);
    const cardsRef = useRef([]);

    useGSAP(() => {
        cardsRef.current.forEach((card, index) => {
            gsap.from(card, {
                scrollTrigger: {
                    trigger: card,
                    start: 'top 85%',
                    toggleActions: 'play none none reverse',
                },
                y: 50,
                opacity: 0,
                duration: 0.8,
                delay: index * 0.2,
                ease: 'power2.out',
            });
        });
    }, { scope: containerRef });

    return (
        <div ref={containerRef} className="py-24 sm:py-32 relative bg-slate-50 dark:bg-[#0f172a] transition-colors duration-300">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="mx-auto max-w-2xl lg:text-center mb-16">
                    <h2 className="text-base font-semibold leading-7 text-sky-500 dark:text-sky-400 tracking-wide uppercase">Capabilities</h2>
                    <p className="mt-2 text-3xl font-bold tracking-tight text-slate-900 dark:text-white sm:text-5xl">
                        Precision Thermal Mapping
                    </p>
                    <p className="mt-6 text-lg leading-8 text-slate-600 dark:text-gray-400">
                        Our model leverages advanced convolutional neural networks to bridge the resolution gap between thermal and optical satellite imagery.
                    </p>
                </div>

                <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
                    {features.map((feature, index) => (
                        <div
                            key={feature.name}
                            ref={el => cardsRef.current[index] = el}
                            className="glass-card rounded-2xl p-8 hover:bg-white/80 dark:hover:bg-white/5 transition-colors duration-300 group"
                        >
                            <div className="h-12 w-12 flex items-center justify-center rounded-xl bg-sky-500/10 group-hover:bg-sky-500/20 transition-colors mb-6">
                                {feature.icon}
                            </div>
                            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-3">
                                {feature.name}
                            </h3>
                            <p className="text-slate-600 dark:text-gray-400 leading-relaxed">
                                {feature.description}
                            </p>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default Features;
