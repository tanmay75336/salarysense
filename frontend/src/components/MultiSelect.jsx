import React, { useState, useRef, useEffect } from 'react';

function MultiSelect({ options, selected, onChange, placeholder, maxVisible = 5 }) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const dropdownRef = useRef(null);
  const toggleRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target) &&
          toggleRef.current && !toggleRef.current.contains(event.target)) {
        setIsOpen(false);
        setSearchTerm('');
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const filteredOptions = options.filter(opt =>
    opt.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleToggle = () => {
    setIsOpen(!isOpen);
    if (!isOpen) setSearchTerm('');
  };

  const handleSelect = (option) => {
    const newSelected = selected.includes(option)
      ? selected.filter(s => s !== option)
      : [...selected, option];
    onChange(newSelected);
  };

  const displayText = selected.length === 0
    ? placeholder
    : selected.length <= maxVisible
    ? selected.join(', ')
    : `${selected.slice(0, maxVisible).join(', ')} +${selected.length - maxVisible} more`;

  return (
    <div className="multi-select" ref={dropdownRef}>
      <div
        ref={toggleRef}
        className="multi-select-toggle"
        onClick={handleToggle}
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); handleToggle(); }}}
        role="combobox"
        aria-expanded={isOpen}
        aria-haspopup="listbox"
      >
        <span className={selected.length === 0 ? 'multi-select-placeholder' : 'multi-select-selected'}>
          {displayText}
        </span>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: 1 }}>
          {isOpen ? '▲' : '▼'}
        </span>
      </div>

      {isOpen && (
        <div className="multi-select-dropdown" role="listbox">
          <input
            type="text"
            placeholder="Search..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onClick={(e) => e.stopPropagation()}
            style={{
              padding: '8px 12px',
              border: 'none',
              borderBottom: '1px solid #DED5C8',
              width: '100%',
              fontSize: '0.85rem',
              outline: 'none',
              background: '#FFFCF7',
              color: '#302B25',
              fontFamily: 'inherit'
            }}
          />
          {filteredOptions.length === 0 ? (
            <div style={{ padding: '14px', textAlign: 'center', color: '#91877B', fontSize: '0.82rem' }}>
              No matching options
            </div>
          ) : (
            filteredOptions.map(option => (
              <label
                key={option}
                className="multi-select-option"
                onClick={(e) => e.stopPropagation()}
              >
                <input
                  type="checkbox"
                  checked={selected.includes(option)}
                  onChange={() => handleSelect(option)}
                  onClick={(e) => e.stopPropagation()}
                />
                <span>{option}</span>
              </label>
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default MultiSelect;