import React, { useState, useRef } from 'react';
import gsap from 'gsap';
import { useGSAP } from '@gsap/react';

const Demo = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [dragActive, setDragActive] = useState(false);

    const containerRef = useRef(null);
    const formRef = useRef(null);
    const resultRef = useRef(null);

    useGSAP(() => {
        gsap.from(formRef.current, {
            scrollTrigger: {
                trigger: formRef.current,
                start: 'top 80%',
            },
            y: 30,
            opacity: 0,
            duration: 1,
            ease: 'power3.out'
        });
    }, { scope: containerRef });

    useGSAP(() => {
        if (result && resultRef.current) {
            gsap.fromTo(resultRef.current,
                { y: 50, opacity: 0 },
                { y: 0, opacity: 1, duration: 0.8, ease: 'back.out(1.2)' }
            );
        }
    }, [result]);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setSelectedFile(e.dataTransfer.files[0]);
            setResult(null);
            setError(null);
        }
    };

    const handleFileChange = (event) => {
        setSelectedFile(event.target.files[0]);
        setResult(null);
        setError(null);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (!selectedFile) return;

        setIsLoading(true);
        setError(null);

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

            if (data.error) {
                throw new Error(data.error);
            }

            setResult(data);

        } catch (err) {
            console.error("Error:", err);
            setError(err.message || "An unexpected error occurred.");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div id="demo" ref={containerRef} className="py-24 sm:py-32 relative bg-slate-50 dark:bg-[#0f172a] transition-colors duration-300">
            <div className="mx-auto max-w-7xl px-6 lg:px-8">
                <div className="mx-auto max-w-2xl text-center mb-16">
                    <h2 className="text-3xl font-bold tracking-tight text-slate-900 dark:text-white sm:text-4xl">Live Demonstration</h2>
                    <p className="mt-4 text-lg leading-8 text-slate-600 dark:text-gray-400">
                        Experience the power of CSRN. Upload a Landsat 8/9 GeoTIFF bundle to see the super-resolution in action.
                    </p>
                </div>

                <div ref={formRef} className="mx-auto max-w-3xl glass rounded-2xl p-8 sm:p-12">
                    <form onSubmit={handleSubmit} className="space-y-8">
                        <div
                            className={`relative flex justify-center rounded-xl border-2 border-dashed px-6 py-16 transition-all duration-300 ${dragActive
                                ? 'border-sky-500 bg-sky-500/10'
                                : 'border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-500 hover:bg-slate-50 dark:hover:bg-slate-800/50'
                                }`}
                            onDragEnter={handleDrag}
                            onDragLeave={handleDrag}
                            onDragOver={handleDrag}
                            onDrop={handleDrop}
                        >
                            <div className="text-center">
                                <div className="mx-auto h-16 w-16 flex items-center justify-center rounded-full bg-slate-100 dark:bg-slate-800 mb-4">
                                    <svg width="32" height="32" className="h-8 w-8 text-sky-500 dark:text-sky-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                    </svg>
                                </div>
                                <div className="mt-4 flex text-sm leading-6 text-slate-600 dark:text-gray-400 justify-center">
                                    <label
                                        htmlFor="file-upload"
                                        className="relative cursor-pointer rounded-md font-semibold text-sky-500 dark:text-sky-400 focus-within:outline-none focus-within:ring-2 focus-within:ring-sky-500 focus-within:ring-offset-2 hover:text-sky-600 dark:hover:text-sky-300"
                                    >
                                        <span>Upload a file</span>
                                        <input id="file-upload" name="file-upload" type="file" className="sr-only" onChange={handleFileChange} accept=".tif,.tiff" />
                                    </label>
                                    <p className="pl-1">or drag and drop</p>
                                </div>
                                <p className="text-xs leading-5 text-slate-500 dark:text-gray-500 mt-2">GeoTIFF up to 50MB</p>
                                {selectedFile && (
                                    <div className="mt-4 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-sky-500/10 text-sky-600 dark:text-sky-400 text-sm">
                                        <svg width="16" height="16" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        {selectedFile.name}
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="flex justify-center">
                            <button
                                type="submit"
                                disabled={!selectedFile || isLoading}
                                className={`
                    flex w-full sm:w-auto justify-center items-center rounded-full px-8 py-3 text-base font-semibold text-white shadow-lg transition-all duration-200
                    ${!selectedFile || isLoading
                                        ? 'bg-slate-400 dark:bg-slate-700 cursor-not-allowed opacity-50'
                                        : 'bg-gradient-to-r from-sky-500 to-indigo-500 hover:from-sky-400 hover:to-indigo-400 hover:shadow-sky-500/25'
                                    }
                `}
                            >
                                {isLoading ? (
                                    <>
                                        <svg width="20" height="20" className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Processing Scene...
                                    </>
                                ) : 'Run Inference'}
                            </button>
                        </div>
                        {error && (
                            <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-600 dark:text-red-400 text-sm text-center">
                                {error}
                            </div>
                        )}
                    </form>
                </div>

                {result && (
                    <div ref={resultRef} className="mt-24">
                        <div className="grid grid-cols-1 gap-12 lg:grid-cols-3">
                            {/* Input Preview */}
                            <div className="glass rounded-2xl p-6 space-y-4">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-lg font-medium text-slate-900 dark:text-white">Input (100m)</h3>
                                    <span className="px-2 py-1 rounded bg-slate-200 dark:bg-slate-800 text-xs text-slate-600 dark:text-gray-400">Original</span>
                                </div>
                                <div className="aspect-square rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-900 relative group">
                                    <img src={result.input_preview} alt="Input" className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" />
                                </div>
                            </div>

                            {/* Output Preview */}
                            <div className="glass rounded-2xl p-6 space-y-4 ring-1 ring-sky-500/50 shadow-[0_0_50px_-12px_rgba(56,189,248,0.25)]">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-lg font-medium text-sky-600 dark:text-sky-400">Prediction (30m)</h3>
                                    <span className="px-2 py-1 rounded bg-sky-500/10 text-xs text-sky-600 dark:text-sky-400">Super-Resolved</span>
                                </div>
                                <div className="aspect-square rounded-xl overflow-hidden border border-sky-500/30 bg-slate-100 dark:bg-slate-900 relative group">
                                    <img src={result.output_preview} alt="Output" className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" />
                                </div>
                            </div>

                            {/* Metrics */}
                            <div className="glass rounded-2xl p-6 flex flex-col justify-center space-y-8">
                                <h3 className="text-lg font-medium text-slate-900 dark:text-white border-b border-slate-200 dark:border-slate-700 pb-4">Performance Metrics</h3>

                                <div className="space-y-6">
                                    <div>
                                        <div className="flex justify-between items-end mb-2">
                                            <p className="text-sm text-slate-600 dark:text-gray-400">RMSE (Root Mean Square Error)</p>
                                            <p className="text-2xl font-bold text-slate-900 dark:text-white">{result.metrics.rmse.toFixed(4)} K</p>
                                        </div>
                                        <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-2">
                                            <div className="bg-emerald-500 h-2 rounded-full" style={{ width: '85%' }}></div>
                                        </div>
                                    </div>

                                    <div>
                                        <div className="flex justify-between items-end mb-2">
                                            <p className="text-sm text-slate-600 dark:text-gray-400">PSNR (Peak Signal-to-Noise Ratio)</p>
                                            <p className="text-2xl font-bold text-slate-900 dark:text-white">{result.metrics.psnr.toFixed(4)} dB</p>
                                        </div>
                                        <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-2">
                                            <div className="bg-sky-500 h-2 rounded-full" style={{ width: '92%' }}></div>
                                        </div>
                                    </div>

                                    <div>
                                        <div className="flex justify-between items-end mb-2">
                                            <p className="text-sm text-slate-600 dark:text-gray-400">SSIM (Structural Similarity)</p>
                                            <p className="text-2xl font-bold text-slate-900 dark:text-white">{result.metrics.ssim.toFixed(4)}</p>
                                        </div>
                                        <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-2">
                                            <div className="bg-indigo-500 h-2 rounded-full" style={{ width: '89%' }}></div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Demo;
