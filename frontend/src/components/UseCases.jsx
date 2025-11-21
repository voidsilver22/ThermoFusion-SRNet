import React, { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const cases = [
    {
        title: 'Urban Heat Islands',
        category: 'City Planning',
        image: 'https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?q=80&w=2070&auto=format&fit=crop',
        desc: 'Identify micro-climates in dense urban areas to optimize green infrastructure.'
    },
    {
        title: 'Precision Agriculture',
        category: 'Farming',
        image: 'https://images.unsplash.com/photo-1625246333195-78d9c38ad449?q=80&w=2070&auto=format&fit=crop',
        desc: 'Monitor crop stress and irrigation efficiency at a sub-field level.'
    },
    {
        title: 'Water Resource Mgmt',
        category: 'Environment',
        image: 'https://images.unsplash.com/photo-1589802829985-817e51171b92?q=80&w=2070&auto=format&fit=crop',
        desc: 'Track evaporation rates and water quality in reservoirs and lakes.'
    },
];

const UseCases = () => {
    const containerRef = useRef(null);

    useGSAP(() => {
        gsap.utils.toArray('.case-card').forEach((card, i) => {
            gsap.from(card, {
                scrollTrigger: {
                    trigger: card,
                    start: 'top 85%',
                    toggleActions: 'play none none reverse',
                },
                y: 100,
                opacity: 0,
                duration: 1,
                delay: i * 0.2,
                ease: 'power4.out',
            });
        });
    }, { scope: containerRef });

    return (
        <div ref={containerRef} className="py-32 relative z-10">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="flex flex-col md:flex-row justify-between items-end mb-16 gap-6">
                    <div>
                        <h2 className="text-base font-semibold leading-7 text-orange-400 tracking-wide uppercase">Applications</h2>
                        <p className="mt-2 text-4xl font-bold tracking-tight text-white sm:text-5xl">
                            Real-World Impact
                        </p>
                    </div>
                    <p className="max-w-md text-lg text-gray-400 text-right">
                        From city streets to crop fields, high-resolution thermal data drives better decisions.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {cases.map((item, index) => (
                        <div key={index} className="case-card group relative h-[400px] rounded-3xl overflow-hidden cursor-pointer">
                            <img
                                src={item.image}
                                alt={item.title}
                                className="absolute inset-0 w-full h-full object-cover transition-transform duration-700 group-hover:scale-110"
                            />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/90 via-black/20 to-transparent opacity-80 group-hover:opacity-90 transition-opacity"></div>

                            <div className="absolute bottom-0 left-0 p-8 w-full transform translate-y-4 group-hover:translate-y-0 transition-transform duration-300">
                                <span className="inline-block px-3 py-1 rounded-full bg-white/10 backdrop-blur-md text-xs font-medium text-white mb-3 border border-white/20">
                                    {item.category}
                                </span>
                                <h3 className="text-2xl font-bold text-white mb-2">{item.title}</h3>
                                <p className="text-gray-300 text-sm opacity-0 group-hover:opacity-100 transition-opacity duration-300 delay-100">
                                    {item.desc}
                                </p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default UseCases;
