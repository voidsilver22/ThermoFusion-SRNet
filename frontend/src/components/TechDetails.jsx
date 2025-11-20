import React from 'react';
import architectureDiagram from '../assets/architecture_diagram.png';

const TechDetails = () => {
    return (
        <div id="tech-details" className="py-24 sm:py-32 relative overflow-hidden">
            {/* Background Decoration */}
            <div className="absolute right-0 top-1/4 w-1/2 h-1/2 bg-indigo-900/20 blur-[100px] rounded-full pointer-events-none"></div>

            <div className="mx-auto max-w-7xl px-6 lg:px-8 relative z-10">
                <div className="mx-auto max-w-2xl lg:mx-0">
                    <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">Technical Architecture</h2>
                    <p className="mt-6 text-lg leading-8 text-gray-300">
                        The Cross-Scale Residual Network (CSRN) utilizes a deep learning approach to fuse multi-modal satellite data.
                    </p>
                </div>

                <div className="mx-auto mt-16 grid max-w-2xl grid-cols-1 gap-x-12 gap-y-16 lg:mx-0 lg:max-w-none lg:grid-cols-2 lg:items-center">
                    <div className="lg:pr-4">
                        <div className="lg:max-w-lg space-y-8">
                            <div>
                                <div className="flex items-center gap-3 mb-4">
                                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-sky-500/10 ring-1 ring-sky-500/20 text-sky-400 font-bold text-sm">1</span>
                                    <h3 className="text-xl font-semibold text-white">The Model</h3>
                                </div>
                                <p className="text-gray-400 pl-11">
                                    Our architecture employs residual learning to capture high-frequency details from optical bands and inject them into the thermal data.
                                </p>
                            </div>

                            <div>
                                <div className="flex items-center gap-3 mb-4">
                                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-sky-500/10 ring-1 ring-sky-500/20 text-sky-400 font-bold text-sm">2</span>
                                    <h3 className="text-xl font-semibold text-white">Data Fusion</h3>
                                </div>
                                <p className="text-gray-400 pl-11">
                                    Input: 7-band Landsat 8/9 OLI data (30m) + single-band TIRS thermal image (100m).
                                    <br />
                                    Output: Super-resolved Land Surface Temperature (30m).
                                </p>
                            </div>

                            <div>
                                <div className="flex items-center gap-3 mb-4">
                                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-sky-500/10 ring-1 ring-sky-500/20 text-sky-400 font-bold text-sm">3</span>
                                    <h3 className="text-xl font-semibold text-white">Optimization</h3>
                                </div>
                                <p className="text-gray-400 pl-11">
                                    Trained using a combination of L1 Loss and Perceptual Loss to ensure both pixel-wise accuracy and texture preservation.
                                </p>
                            </div>
                        </div>
                    </div>

                    <div className="glass rounded-2xl p-2 ring-1 ring-white/10">
                        <div className="aspect-[16/9] bg-slate-900/50 rounded-xl overflow-hidden relative flex items-center justify-center group">
                            <img
                                src={architectureDiagram}
                                alt="CSRN Architecture Diagram"
                                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
                            />
                            <div className="absolute inset-0 bg-gradient-to-t from-slate-900/80 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-end justify-center pb-4">
                                <span className="text-white font-medium">CSRN Architecture Overview</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TechDetails;
