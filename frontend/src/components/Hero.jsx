import React, { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';

const Hero = () => {
    const containerRef = useRef(null);
    const titleRef = useRef(null);

    useGSAP(() => {
        const tl = gsap.timeline({ defaults: { ease: 'power4.out' } });

        tl.from('.hero-word', {
            y: 150,
            opacity: 0,
            duration: 1.5,
            stagger: 0.1,
            rotate: 5,
            transformOrigin: 'left top',
        })
            .from('.hero-sub', {
                y: 20,
                opacity: 0,
                duration: 1,
            }, '-=1')
            .from('.hero-btn', {
                scale: 0.8,
                opacity: 0,
                duration: 1,
                ease: 'elastic.out(1, 0.5)',
            }, '-=0.8');

    }, { scope: containerRef });

    return (
        <div ref={containerRef} className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden pt-20">

            <div className="relative z-10 text-center px-4 w-full max-w-[90vw]">
                {/* Massive Typography */}
                <h1 ref={titleRef} className="font-bold leading-[0.85] tracking-tighter text-[#fffce1]">
                    <div className="overflow-hidden">
                        <span className="hero-word inline-block text-[15vw] xl:text-[18vw]">Thermo</span>
                    </div>
                    <div className="overflow-hidden">
                        <span className="hero-word inline-block text-[15vw] xl:text-[18vw] text-transparent bg-clip-text bg-gradient-to-r from-purple-400 via-pink-500 to-orange-500">Fusion</span>
                    </div>
                </h1>

                <p className="hero-sub mt-8 text-xl md:text-2xl text-gray-400 max-w-2xl mx-auto font-light tracking-wide">
                    Super-resolution thermal imaging powered by <span className="text-white font-medium">CSRN</span>.
                </p>

                <div className="mt-12 flex justify-center gap-6">
                    <a href="#demo" className="hero-btn group relative inline-flex items-center justify-center px-8 py-4 text-lg font-bold text-[#0e100f] transition-all duration-200 bg-[#fffce1] rounded-full hover:scale-105 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-white">
                        Start Analyzing
                        <div className="absolute inset-0 rounded-full ring-2 ring-white/50 group-hover:ring-white/80 scale-110 opacity-0 group-hover:opacity-100 transition-all duration-300"></div>
                    </a>
                </div>
            </div>

            {/* Decorative "Scroll" Indicator */}
            <div className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 opacity-50 animate-bounce">
                <span className="text-xs uppercase tracking-widest">Scroll</span>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 5v14M19 12l-7 7-7-7" />
                </svg>
            </div>
        </div>
    );
};

export default Hero;
