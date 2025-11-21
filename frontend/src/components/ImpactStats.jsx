import React, { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const stats = [
    { label: 'Resolution Increase', value: '3x', desc: 'From 100m to 30m' },
    { label: 'Model Accuracy', value: '95%', desc: 'SSIM Score > 0.95' },
    { label: 'Processing Speed', value: '<2s', desc: 'Per Landsat Scene' },
    { label: 'Data Points', value: '1M+', desc: 'Pixels Enhanced' },
];

const ImpactStats = () => {
    const containerRef = useRef(null);

    useGSAP(() => {
        gsap.from('.stat-item', {
            scrollTrigger: {
                trigger: containerRef.current,
                start: 'top 80%',
                toggleActions: 'play none none reverse',
            },
            y: 50,
            opacity: 0,
            duration: 1,
            stagger: 0.1,
            ease: 'power3.out',
        });
    }, { scope: containerRef });

    return (
        <div ref={containerRef} className="py-24 relative z-10 backdrop-blur-sm border-y border-white/5">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="grid grid-cols-2 gap-8 lg:grid-cols-4 text-center">
                    {stats.map((stat, index) => (
                        <div key={index} className="stat-item flex flex-col gap-y-2">
                            <dt className="text-sm leading-6 text-gray-400">{stat.label}</dt>
                            <dd className="order-first text-5xl font-bold tracking-tight text-white">
                                {stat.value}
                            </dd>
                            <span className="text-xs text-gray-500">{stat.desc}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default ImpactStats;
