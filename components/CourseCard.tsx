
import React from 'react';
import { Course } from '../types';

interface CourseCardProps {
  course: Course;
  onStart: (course: Course) => void;
  progress?: number;
}

const CourseCard: React.FC<CourseCardProps> = ({ course, onStart, progress = 0 }) => {
  const difficultyColor = {
    'Beginner': 'text-green-400 bg-green-400/10',
    'Intermediate': 'text-yellow-400 bg-yellow-400/10',
    'Advanced': 'text-red-400 bg-red-400/10',
  };

  return (
    <div 
      className="bg-slate-800/50 border border-slate-700 rounded-xl p-6 hover:border-blue-500/50 transition-all group cursor-pointer flex flex-col h-full"
      onClick={() => onStart(course)}
    >
      <div className="flex justify-between items-start mb-4">
        <span className="px-3 py-1 text-xs font-semibold rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20">
          {course.category}
        </span>
        <span className={`px-2 py-0.5 text-[10px] uppercase tracking-wider font-bold rounded ${difficultyColor[course.difficulty]}`}>
          {course.difficulty}
        </span>
      </div>
      <h3 className="text-xl font-bold text-white mb-2 group-hover:text-blue-400 transition-colors">
        {course.title}
      </h3>
      <p className="text-slate-400 text-sm mb-6 line-clamp-2 flex-grow">
        {course.description}
      </p>

      {/* Progress Bar */}
      {progress > 0 && (
        <div className="mb-6">
          <div className="flex justify-between items-center mb-1.5">
            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Progress</span>
            <span className="text-[10px] font-bold text-blue-400">{progress}%</span>
          </div>
          <div className="w-full h-1 bg-slate-900 rounded-full overflow-hidden">
            <div 
              className="h-full bg-blue-500 transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      <div className="flex items-center justify-between text-xs text-slate-500 font-medium pt-4 border-t border-slate-700 mt-auto">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {course.duration}
          </span>
          <span className="flex items-center gap-1">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
            {course.lessons} Lessons
          </span>
        </div>
        <button className="text-blue-400 hover:text-blue-300 font-bold flex items-center gap-1">
          {progress === 100 ? 'Review' : progress > 0 ? 'Continue' : 'Start'}
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default CourseCard;
