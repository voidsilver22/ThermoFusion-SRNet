import React, { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const MorphingShapes = () => {
    const containerRef = useRef(null);

    useGSAP(() => {
        const shapes = gsap.utils.toArray('.morph-shape');

        shapes.forEach((shape, i) => {
            // Random initial position
            gsap.set(shape, {
                x: gsap.utils.random(0, window.innerWidth),
                y: gsap.utils.random(0, window.innerHeight),
                rotation: gsap.utils.random(0, 360),
                scale: gsap.utils.random(0.5, 1.5),
            });

            // Continuous floating animation
            gsap.to(shape, {
                x: `+=${gsap.utils.random(-100, 100)}`,
                y: `+=${gsap.utils.random(-100, 100)}`,
                rotation: `+=${gsap.utils.random(-180, 180)}`,
                duration: gsap.utils.random(10, 20),
                repeat: -1,
                yoyo: true,
                ease: 'sine.inOut',
            });

            // Scroll-linked movement (Parallax)
            gsap.to(shape, {
                y: `+=${gsap.utils.random(200, 500)}`, // Move down with scroll
                ease: 'none',
                scrollTrigger: {
                    trigger: document.body,
                    start: 'top top',
                    end: 'bottom bottom',
                    scrub: 1.5,
                }
            });
        });

    }, { scope: containerRef });

    return (
        <div ref={containerRef} className="fixed inset-0 pointer-events-none overflow-hidden z-10">
            <svg className="absolute w-0 h-0">
                <defs>
                    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#d946ef" />
                        <stop offset="100%" stopColor="#8b5cf6" />
                    </linearGradient>
                    <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#0ae448" />
                        <stop offset="100%" stopColor="#06b6d4" />
                    </linearGradient>
                    <linearGradient id="grad3" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#f59e0b" />
                        <stop offset="100%" stopColor="#ec4899" />
                    </linearGradient>
                </defs>
            </svg>

            {/* Organic Blob 1 - Purple/Violet Gradient */}
            <svg className="morph-shape absolute w-96 h-96 opacity-70 blur-[60px]" viewBox="0 0 200 200">
                <path fill="url(#grad1)" d="M44.7,-76.4C58.9,-69.2,71.8,-59.1,81.6,-46.6C91.4,-34.1,98.1,-19.2,95.8,-5.3C93.5,8.6,82.2,21.5,71.1,32.6C60,43.7,49.1,53,37.2,60.3C25.3,67.6,12.4,72.9,-0.6,73.9C-13.6,74.9,-27.8,71.6,-40.3,64.2C-52.8,56.8,-63.6,45.3,-71.3,32.1C-79,18.9,-83.6,4,-81.6,-9.8C-79.6,-23.6,-71,-36.3,-60.1,-45.8C-49.2,-55.3,-36,-61.6,-22.7,-69.3C-9.4,-77,4,-86.1,17.3,-86.3C30.6,-86.5,43.8,-77.8,44.7,-76.4Z" transform="translate(100 100)" />
            </svg>

            {/* Organic Blob 2 - Green/Cyan Gradient */}
            <svg className="morph-shape absolute w-[500px] h-[500px] opacity-60 blur-[80px]" viewBox="0 0 200 200">
                <path fill="url(#grad2)" d="M41.7,-73.4C54.6,-64.8,66.1,-55.2,75.3,-43.6C84.5,-32,91.4,-18.4,90.4,-5.2C89.4,8,80.5,20.8,70.6,31.7C60.7,42.6,49.8,51.6,38.2,58.8C26.6,66,14.3,71.4,1.4,69C-11.5,66.6,-25,56.4,-37.4,47.2C-49.8,38,-61.1,29.8,-68.8,18.3C-76.5,6.8,-80.6,-8,-76.7,-21.3C-72.8,-34.6,-60.9,-46.4,-48.3,-55.1C-35.7,-63.8,-22.4,-69.4,-9.3,-69.2C3.8,-69,17,-63,28.8,-82" transform="translate(100 100)" />
            </svg>

            {/* Geometric Shape 1 (Triangle-ish) - Orange/Pink Gradient */}
            <svg className="morph-shape absolute w-64 h-64 opacity-70 mix-blend-screen" viewBox="0 0 200 200">
                <path fill="url(#grad3)" d="M100 20 L180 180 L20 180 Z" />
            </svg>

            {/* Geometric Shape 2 (Circle) - Gradient Border */}
            <div className="morph-shape absolute w-48 h-48 rounded-full border-4 border-transparent opacity-80 shadow-[0_0_30px_rgba(236,72,153,0.5)]"
                style={{ background: 'linear-gradient(#0e100f, #0e100f) padding-box, linear-gradient(to right, #ec4899, #8b5cf6) border-box' }}>
            </div>

            {/* Geometric Shape 3 (Square) - Glassy Blue */}
            <div className="morph-shape absolute w-32 h-32 rotate-45 backdrop-blur-md bg-gradient-to-br from-blue-500/40 to-cyan-400/40 border border-white/20 shadow-[0_0_20px_rgba(59,130,246,0.6)]"></div>

            {/* More Blobs for density - Mixed Gradient */}
            <svg className="morph-shape absolute w-80 h-80 opacity-50 blur-[50px]" viewBox="0 0 200 200">
                <path fill="url(#grad1)" d="M38.1,-63.8C49.6,-54.6,59.3,-44.2,67.6,-32.4C75.9,-20.6,82.8,-7.4,81.3,5.1C79.8,17.6,69.9,29.4,59.6,39.6C49.3,49.8,38.6,58.4,26.9,64.2C15.2,70,2.5,73,-9.8,71.8C-22.1,70.6,-34,65.2,-44.6,57.2C-55.2,49.2,-64.5,38.6,-70.6,26.3C-76.7,14,-79.6,0,-76.9,-12.8C-74.2,-25.6,-65.9,-37.2,-55.3,-46.8C-44.7,-56.4,-31.8,-64,-19.1,-66.2C-6.4,-68.4,6.1,-65.2,18.1,-61.8" transform="translate(100 100)" />
            </svg>
        </div>
    );
};

export default MorphingShapes;
