import React, { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const steps = [
    {
        num: '01',
        title: 'Ingest',
        desc: 'Landsat 8/9 OLI (30m) and TIRS (100m) bands are ingested and preprocessed.',
    },
    {
        num: '02',
        title: 'Extract',
        desc: 'Deep feature extraction via Residual Blocks captures high-frequency details from optical data.',
    },
    {
        num: '03',
        title: 'Fuse',
        desc: 'Cross-scale attention mechanisms fuse thermal and optical features.',
    },
    {
        num: '04',
        title: 'Refine',
        desc: 'Upsampling layers reconstruct the thermal map at 30m resolution.',
    },
];

const ProcessFlow = () => {
    const containerRef = useRef(null);
    const lineRef = useRef(null);

    useGSAP(() => {
        // Animate the connecting line
        gsap.fromTo(lineRef.current,
            { height: '0%' },
            {
                height: '100%',
                ease: 'none',
                scrollTrigger: {
                    trigger: containerRef.current,
                    start: 'top center',
                    end: 'bottom center',
                    scrub: 1,
                }
            }
        );

        // Animate steps
        gsap.utils.toArray('.process-step').forEach((step) => {
            gsap.from(step, {
                scrollTrigger: {
                    trigger: step,
                    start: 'top 80%',
                    toggleActions: 'play none none reverse',
                },
                x: -50,
                opacity: 0,
                duration: 0.8,
                ease: 'power3.out',
            });
        });

    }, { scope: containerRef });

    return (
        <div className="py-32 relative overflow-hidden">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="mx-auto max-w-2xl lg:text-center mb-20">
                    <h2 className="text-base font-semibold leading-7 text-purple-400 tracking-wide uppercase">How It Works</h2>
                    <p className="mt-2 text-4xl font-bold tracking-tight text-white sm:text-5xl">
                        The CSRN Pipeline
                    </p>
                </div>

                <div ref={containerRef} className="relative max-w-3xl mx-auto pl-8 lg:pl-0">
                    {/* Connecting Line */}
                    <div className="absolute left-8 lg:left-1/2 top-0 bottom-0 w-0.5 bg-white/10 -translate-x-1/2">
                        <div ref={lineRef} className="w-full bg-gradient-to-b from-purple-500 to-orange-500"></div>
                    </div>

                    <div className="space-y-24">
                        {steps.map((step, index) => (
                            <div key={index} className={`process-step relative flex flex-col lg:flex-row gap-8 items-center ${index % 2 === 0 ? 'lg:flex-row-reverse' : ''}`}>

                                {/* Content */}
                                <div className={`flex-1 ${index % 2 === 0 ? 'lg:text-left' : 'lg:text-right'} text-left`}>
                                    <h3 className="text-3xl font-bold text-white mb-2">{step.title}</h3>
                                    <p className="text-gray-400 text-lg leading-relaxed">{step.desc}</p>
                                </div>

                                {/* Center Marker */}
                                <div className="absolute left-0 lg:left-1/2 -translate-x-1/2 w-16 h-16 rounded-full bg-[#0e100f] border-4 border-white/10 flex items-center justify-center z-10">
                                    <span className="text-xl font-bold text-white">{step.num}</span>
                                </div>

                                {/* Empty space for balance */}
                                <div className="flex-1 hidden lg:block"></div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ProcessFlow;
