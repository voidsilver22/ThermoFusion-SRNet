import React, { useState, useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const Demo = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [result, setResult] = useState(null);
    const containerRef = useRef(null);

    useGSAP(() => {
        gsap.from('.demo-content', {
            scrollTrigger: {
                trigger: containerRef.current,
                start: 'top 80%',
            },
            y: 50,
            opacity: 0,
            duration: 1,
            ease: 'power3.out',
        });
    }, { scope: containerRef });

    const handleFileChange = (e) => {
        setSelectedFile(e.target.files[0]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!selectedFile) return;

        setIsProcessing(true);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to process image');
            }

            setResult({
                original: data.input_preview, // Backend returns 'input_preview'
                enhanced: data.output_preview, // Backend returns 'output_preview'
            });

        } catch (err) {
            console.error("Error:", err);
            // Ideally handle error state here
        } finally {
            setIsProcessing(false);
        }
    };

    return (
        <div id="demo" ref={containerRef} className="py-24 relative">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="demo-content text-center max-w-2xl mx-auto mb-16">
                    <h2 className="text-[#0ae448] font-mono text-sm tracking-wider mb-4 uppercase">Live Demo</h2>
                    <h3 className="text-4xl font-bold text-white mb-6">Experience the Difference</h3>
                    <p className="text-gray-400 text-lg">
                        Upload a sample Landsat thermal band (Band 10/11) and see the CSRN model enhance it in real-time.
                    </p>
                </div>

                <div className="demo-content max-w-4xl mx-auto bg-white/5 border border-white/10 rounded-3xl p-8 md:p-12 backdrop-blur-sm shadow-2xl">
                    <form onSubmit={handleSubmit} className="space-y-8">
                        <div className="flex flex-col items-center justify-center w-full">
                            <label htmlFor="dropzone-file" className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed border-white/20 rounded-2xl cursor-pointer bg-black/20 hover:bg-white/5 transition-colors duration-300 group">
                                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                    <svg className="w-12 h-12 mb-4 text-gray-400 group-hover:text-[#0ae448] transition-colors" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                                        <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2" />
                                    </svg>
                                    <p className="mb-2 text-sm text-gray-400"><span className="font-semibold text-white">Click to upload</span> or drag and drop</p>
                                    <p className="text-xs text-gray-500">GeoTIFF, TIF (MAX. 50MB)</p>
                                </div>
                                <input id="dropzone-file" type="file" className="hidden" onChange={handleFileChange} accept=".tif,.tiff" />
                            </label>
                            {selectedFile && (
                                <p className="mt-4 text-sm text-[#0ae448] font-medium">
                                    Selected: {selectedFile.name}
                                </p>
                            )}
                        </div>

                        <div className="flex justify-center">
                            <button
                                type="submit"
                                disabled={!selectedFile || isProcessing}
                                className={`px-8 py-4 rounded-full font-bold text-black transition-all duration-300 ${!selectedFile || isProcessing
                                    ? 'bg-gray-600 cursor-not-allowed'
                                    : 'bg-[#0ae448] hover:bg-[#0ae448]/90 hover:scale-105 shadow-[0_0_20px_rgba(10,228,72,0.3)]'
                                    }`}
                            >
                                {isProcessing ? (
                                    <span className="flex items-center gap-2">
                                        <svg className="animate-spin h-5 w-5 text-black" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Processing...
                                    </span>
                                ) : (
                                    'Enhance Image'
                                )}
                            </button>
                        </div>
                    </form>

                    {result && (
                        <div className="mt-16 grid grid-cols-1 md:grid-cols-2 gap-8">
                            <div className="space-y-4">
                                <h4 className="text-white font-semibold text-center">Original (100m)</h4>
                                <div className="aspect-square rounded-2xl overflow-hidden border border-white/10 bg-black/50">
                                    <img src={result.original} alt="Original" className="w-full h-full object-cover opacity-80" />
                                </div>
                            </div>
                            <div className="space-y-4">
                                <h4 className="text-[#0ae448] font-semibold text-center">Enhanced (30m)</h4>
                                <div className="aspect-square rounded-2xl overflow-hidden border border-[#0ae448]/30 bg-black/50 relative group">
                                    <div className="absolute inset-0 bg-[#0ae448]/10 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none"></div>
                                    <img src={result.enhanced} alt="Enhanced" className="w-full h-full object-cover" />
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Demo;
