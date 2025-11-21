import React from 'react';
import Hero from './components/Hero';
import Features from './components/Features';
import ImpactStats from './components/ImpactStats';
import ProcessFlow from './components/ProcessFlow';
import UseCases from './components/UseCases';
import TechDetails from './components/TechDetails';
import Demo from './components/Demo';
import SmoothScroll from './components/SmoothScroll';
import ScrollBackground from './components/ScrollBackground';
import MorphingShapes from './components/MorphingShapes';

function App() {
  return (
    <>
      <ScrollBackground />
      <MorphingShapes />
      <SmoothScroll>
        <div className="min-h-screen text-[#fffce1] transition-colors duration-300 relative z-20 selection:bg-[#0ae448] selection:text-black">
          <Hero />
          <ImpactStats />
          <Features />
          <ProcessFlow />
          <TechDetails />
          <UseCases />
          <Demo />

          <footer className="border-t border-white/5 bg-[#0e100f]/50 backdrop-blur-lg py-12 mt-24 relative z-10">
            <div className="mx-auto max-w-7xl px-6 lg:px-8 flex flex-col md:flex-row justify-between items-center gap-6">
              <div className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 flex items-center justify-center font-bold text-white">
                  TF
                </div>
                <span className="font-semibold text-gray-200">ThermoFusion-SRNet</span>
              </div>
              <p className="text-sm text-gray-500">
                &copy; 2025 Research Project. All rights reserved.
              </p>
            </div>
          </footer>
        </div>
      </SmoothScroll>
    </>
  );
}

export default App;
