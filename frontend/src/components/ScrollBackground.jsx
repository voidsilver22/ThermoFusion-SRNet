import React, { useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const ScrollBackground = () => {
    const bgRef = useRef(null);

    useGSAP(() => {
        // Define gradient stops or colors for different sections
        // We'll animate the background color of the container
        const colors = [
            '#0e100f', // Deep Black (Hero)
            '#1a1c2e', // Dark Blue/Purple (Impact)
            '#0f172a', // Slate 900 (Features)
            '#052e16', // Dark Green (Process - hint of nature/thermal)
            '#0e100f'  // Back to Black (Footer)
        ];

        // Create a timeline linked to total scroll distance
        const tl = gsap.timeline({
            scrollTrigger: {
                trigger: document.body,
                start: 'top top',
                end: 'bottom bottom',
                scrub: 1,
            }
        });

        // Animate through the colors
        colors.forEach((color, i) => {
            if (i < colors.length - 1) {
                tl.to(bgRef.current, {
                    backgroundColor: colors[i + 1],
                    duration: 1,
                    ease: 'none'
                });
            }
        });

    }, { scope: bgRef });

    return (
        <div
            ref={bgRef}
            className="fixed inset-0 z-0 transition-colors duration-500 pointer-events-none"
            style={{ backgroundColor: '#0e100f' }}
        />
    );
};

export default ScrollBackground;
