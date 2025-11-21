import React, { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';

const FloatingShapes = () => {
    const containerRef = useRef(null);

    useGSAP(() => {
        const shapes = gsap.utils.toArray('.shape');

        shapes.forEach((shape, i) => {
            gsap.to(shape, {
                y: 'random(-50, 50)',
                x: 'random(-50, 50)',
                rotation: 'random(-180, 180)',
                duration: 'random(3, 6)',
                repeat: -1,
                yoyo: true,
                ease: 'sine.inOut',
                delay: i * 0.5,
            });
        });
    }, { scope: containerRef });

    return (
        <div ref={containerRef} className="absolute inset-0 overflow-hidden pointer-events-none z-0">
            {/* Gradient Blob 1 */}
            <div className="shape absolute top-[10%] left-[10%] w-64 h-64 rounded-full bg-gradient-to-br from-purple-500/30 to-pink-500/30 blur-3xl mix-blend-screen"></div>

            {/* Gradient Blob 2 */}
            <div className="shape absolute top-[40%] right-[15%] w-96 h-96 rounded-full bg-gradient-to-tr from-sky-500/20 to-emerald-500/20 blur-3xl mix-blend-screen"></div>

            {/* Gradient Blob 3 */}
            <div className="shape absolute bottom-[10%] left-[20%] w-80 h-80 rounded-full bg-gradient-to-t from-orange-500/20 to-yellow-500/20 blur-3xl mix-blend-screen"></div>

            {/* 3D-ish Elements (CSS only) */}
            <div className="shape absolute top-[20%] right-[30%] w-24 h-24 border-4 border-white/10 rounded-2xl rotate-12 backdrop-blur-sm"></div>
            <div className="shape absolute bottom-[30%] left-[5%] w-16 h-16 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-xl rotate-45 opacity-60 shadow-lg shadow-indigo-500/20"></div>
        </div>
    );
};

export default FloatingShapes;
