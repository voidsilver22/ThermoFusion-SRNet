import React from 'react';
import Hero from './components/Hero';
import Features from './components/Features';
import TechDetails from './components/TechDetails';
import Demo from './components/Demo';
import ThemeToggle from './components/ThemeToggle';

function App() {
  return (
    <div className="bg-slate-50 dark:bg-[#0f172a] min-h-screen transition-colors duration-300">
      <ThemeToggle />
      <Hero />
      <Features />
      <TechDetails />
      <Demo />

      <footer className="border-t border-white/5 bg-slate-900/50 backdrop-blur-lg py-12 mt-24">
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
  );
}

export default App;
