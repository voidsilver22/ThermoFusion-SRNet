import React, { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const features = [
    {
        title: 'Super Resolution',
        description: 'Upscale 100m thermal bands to 30m with preservation of radiometric integrity.',
        icon: (
            <svg className="w-8 h-8 text-[#0ae448]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
            </svg>
        ),
    },
    {
        title: 'Cross-Scale Fusion',
        description: 'Intelligent merging of Landsat OLI (30m) features with TIRS thermal data.',
        icon: (
            <svg className="w-8 h-8 text-[#9d95ff]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
            </svg>
        ),
    },
    {
        title: 'Real-Time Processing',
        description: 'Optimized inference pipeline delivering results in seconds per scene.',
        icon: (
            <svg className="w-8 h-8 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
        ),
    },
];

const Features = () => {
    const sectionRef = useRef(null);
    const triggerRef = useRef(null);

    useGSAP(() => {
        const pin = gsap.fromTo(
            sectionRef.current,
            { translateX: 0 },
            {
                translateX: '-200vw',
                ease: 'none',
                duration: 1,
                scrollTrigger: {
                    trigger: triggerRef.current,
                    start: 'top top',
                    end: '2000 top',
                    scrub: 0.6,
                    pin: true,
                },
            }
        );
        return () => {
            pin.kill();
        };
    }, { scope: triggerRef });

    return (
        <div ref={triggerRef} className="overflow-hidden">
            <div ref={sectionRef} className="h-screen w-[300vw] flex flex-row relative">

                {/* Intro Section */}
                <div className="w-screen h-full flex items-center justify-center px-12 border-r border-white/5">
                    <div className="max-w-2xl">
                        <h2 className="text-[#0ae448] font-mono text-sm tracking-wider mb-4 uppercase">Capabilities</h2>
                        <p className="text-5xl md:text-7xl font-bold text-white mb-8 leading-tight">
                            Beyond <br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#0ae448] to-[#9d95ff]">
                                Physical Limits.
                            </span>
                        </p>
                        <p className="text-xl text-gray-400 max-w-lg">
                            Our CSRN architecture hallucinates plausible high-frequency details by learning from the optical spectrum.
                        </p>
                    </div>
                </div>

                {/* Feature Cards */}
                {features.map((feature, index) => (
                    <div key={index} className="w-screen h-full flex items-center justify-center px-4 border-r border-white/5 bg-transparent">
                        <div className="group relative w-full max-w-md aspect-[4/5] bg-white/5 rounded-3xl p-10 border border-white/10 hover:border-[#0ae448]/50 transition-colors duration-500 flex flex-col justify-between overflow-hidden">

                            {/* Hover Gradient */}
                            <div className="absolute inset-0 bg-gradient-to-br from-[#0ae448]/20 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>

                            <div className="relative z-10">
                                <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center mb-8 group-hover:scale-110 transition-transform duration-300">
                                    {feature.icon}
                                </div>
                                <h3 className="text-3xl font-bold text-white mb-4">{feature.title}</h3>
                                <p className="text-gray-400 text-lg leading-relaxed">{feature.description}</p>
                            </div>

                            <div className="relative z-10 flex items-center gap-2 text-[#0ae448] font-medium opacity-0 transform translate-y-4 group-hover:opacity-100 group-hover:translate-y-0 transition-all duration-300">
                                <span>Learn more</span>
                                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                                </svg>
                            </div>
                        </div>
                    </div>
                ))}

            </div>
        </div>
    );
};

export default Features;
