import React, { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';

const Hero = () => {
    const containerRef = useRef(null);
    const titleRef = useRef(null);
    const textRef = useRef(null);
    const buttonsRef = useRef(null);
    const visualRef = useRef(null);

    useGSAP(() => {
        const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });

        tl.from(titleRef.current, {
            y: 100,
            opacity: 0,
            duration: 1,
            delay: 0.2,
        })
            .from(textRef.current, {
                y: 50,
                opacity: 0,
                duration: 0.8,
            }, '-=0.6')
            .from(buttonsRef.current, {
                y: 30,
                opacity: 0,
                duration: 0.8,
            }, '-=0.6')
            .from(visualRef.current, {
                x: 100,
                opacity: 0,
                duration: 1.2,
                ease: 'power2.out',
            }, '-=1');

        // Floating animation for background elements
        gsap.to('.floating-orb', {
            y: -20,
            duration: 3,
            repeat: -1,
            yoyo: true,
            ease: 'sine.inOut',
            stagger: 1.5,
        });

    }, { scope: containerRef });

    return (
        <div ref={containerRef} className="relative min-h-screen flex items-center justify-center overflow-hidden pt-20 bg-slate-50 dark:bg-[#0f172a] transition-colors duration-300">
            {/* Background Elements */}
            <div className="floating-orb absolute top-0 left-1/4 w-96 h-96 bg-sky-500/10 dark:bg-sky-500/20 rounded-full blur-3xl"></div>
            <div className="floating-orb absolute bottom-0 right-1/4 w-96 h-96 bg-indigo-500/10 dark:bg-indigo-500/20 rounded-full blur-3xl"></div>

            <div className="relative z-10 max-w-7xl mx-auto px-6 lg:px-8 grid lg:grid-cols-2 gap-12 items-center">
                <div className="text-left space-y-8">
                    <div className="inline-flex items-center space-x-2 bg-white/80 dark:bg-white/5 rounded-full px-4 py-1.5 border border-slate-200 dark:border-white/10 backdrop-blur-sm shadow-sm">
                        <span className="flex h-2 w-2 rounded-full bg-sky-500 dark:bg-sky-400 animate-pulse"></span>
                        <span className="text-sm font-medium text-slate-600 dark:text-sky-200">v1.0 Now Live</span>
                    </div>

                    <div ref={titleRef} className="overflow-hidden">
                        <h1 className="text-5xl lg:text-7xl font-bold tracking-tight text-slate-900 dark:text-white leading-tight">
                            See the Invisible <br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-sky-500 to-indigo-500 dark:from-sky-400 dark:to-indigo-400">
                                in High Resolution
                            </span>
                        </h1>
                    </div>

                    <p ref={textRef} className="text-xl text-slate-600 dark:text-gray-300 max-w-2xl leading-relaxed">
                        Transform 100m Landsat thermal data into 30m precision insights.
                        Powered by the Cross-Scale Residual Network (CSRN) for superior environmental monitoring.
                    </p>

                    <div ref={buttonsRef} className="flex flex-wrap gap-4">
                        <a
                            href="#demo"
                            className="group relative inline-flex items-center justify-center px-8 py-3 text-base font-semibold text-white transition-all duration-200 bg-sky-500 rounded-full hover:bg-sky-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-sky-500 shadow-[0_0_20px_rgba(56,189,248,0.3)] hover:shadow-[0_0_30px_rgba(56,189,248,0.5)]"
                        >
                            Try Live Demo
                            <svg width="20" height="20" className="w-5 h-5 ml-2 -mr-1 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                            </svg>
                        </a>
                        <a
                            href="#tech-details"
                            className="inline-flex items-center justify-center px-8 py-3 text-base font-semibold text-slate-600 dark:text-gray-300 transition-all duration-200 bg-white/80 dark:bg-white/5 border border-slate-200 dark:border-white/10 rounded-full hover:bg-white dark:hover:bg-white/10 hover:text-slate-900 dark:hover:text-white backdrop-blur-sm shadow-sm"
                        >
                            View Architecture
                        </a>
                    </div>
                </div>

                <div ref={visualRef} className="relative lg:h-[600px] flex items-center justify-center">
                    <div className="relative w-full max-w-lg aspect-square">
                        {/* Abstract Visualization */}
                        <div className="absolute inset-0 bg-gradient-to-tr from-sky-500/20 to-indigo-500/20 dark:from-sky-500/30 dark:to-indigo-500/30 rounded-2xl rotate-6 blur-lg"></div>
                        <div className="relative w-full h-full glass rounded-2xl border border-slate-200/50 dark:border-white/10 overflow-hidden shadow-2xl flex items-center justify-center group">
                            <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop')] bg-cover bg-center opacity-20 dark:opacity-50 group-hover:scale-110 transition-transform duration-700"></div>
                            <div className="absolute inset-0 bg-gradient-to-t from-white/80 via-transparent to-transparent dark:from-slate-900 dark:via-transparent dark:to-transparent"></div>
                            <div className="relative text-center p-8">
                                <p className="text-sky-600 dark:text-sky-300 font-mono text-sm mb-2">PROCESSING TILE: 0007999</p>
                                <div className="text-4xl font-bold text-slate-900 dark:text-white mb-1">30m</div>
                                <div className="text-slate-500 dark:text-gray-400 text-sm uppercase tracking-widest">Resolution Achieved</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Hero;
