
import React, { useState } from 'react';

interface HeaderProps {
  onSearch: (query: string) => void;
  onNavigate: (target: 'home' | 'courses' | 'playground') => void;
}

const Header: React.FC<HeaderProps> = ({ onSearch, onNavigate }) => {
  const [searchValue, setSearchValue] = useState('');

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchValue(e.target.value);
    onSearch(e.target.value);
  };

  return (
    <header className="sticky top-0 z-40 w-full bg-slate-950/80 backdrop-blur-md border-b border-slate-800">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div 
            className="flex items-center gap-3 shrink-0 cursor-pointer group"
            onClick={() => onNavigate('home')}
          >
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-600 to-emerald-500 flex items-center justify-center font-bold text-white text-xl shadow-lg shadow-blue-600/20 group-hover:scale-110 transition-transform">
              D
            </div>
            <span className="hidden sm:block text-xl font-bold tracking-tighter text-white">DevAI</span>
          </div>

          {/* Search */}
          <div className="flex-1 max-w-md mx-8">
            <div className="relative group">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <svg className="h-4 w-4 text-slate-500 group-focus-within:text-blue-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>
              <input
                type="text"
                value={searchValue}
                onChange={handleSearchChange}
                placeholder="Search topics, models, libraries..."
                className="block w-full pl-10 pr-3 py-2 border border-slate-800 rounded-xl bg-slate-900 text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-blue-500 text-sm transition-all"
              />
            </div>
          </div>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            <button 
              onClick={() => onNavigate('courses')}
              className="text-sm font-medium text-slate-400 hover:text-white transition-colors"
            >
              Paths
            </button>
            <button 
              onClick={() => onNavigate('playground')}
              className="text-sm font-medium text-slate-400 hover:text-white transition-colors"
            >
              Library
            </button>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-sm font-medium text-slate-400 hover:text-white transition-colors">
              Community
            </a>
            <button className="bg-slate-800 hover:bg-slate-700 text-white px-5 py-2 rounded-lg text-sm font-bold transition-all border border-slate-700">
              Sign In
            </button>
          </nav>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button className="text-slate-400 hover:text-white p-2">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16m-7 6h7" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
