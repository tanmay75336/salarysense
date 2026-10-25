import React from 'react';

function DatasetInfo({ datasetInfo }) {
  if (!datasetInfo) {
    return <div className="loading">Loading dataset info...</div>;
  }

  return (
    <div className="card">
      <h2>Dataset Information</h2>
      
      <div className="grid-2">
        <div>
          <h3>Basic Info</h3>
          <p><strong>Rows:</strong> {datasetInfo.row_count.toLocaleString()}</p>
          <p><strong>Columns:</strong> {datasetInfo.column_count}</p>
          <p><strong>Type:</strong> {datasetInfo.is_synthetic ? 'Synthetic (Educational)' : 'Real-world'}</p>
          <p><strong>Source:</strong> {datasetInfo.source}</p>
        </div>
        <div>
          <h3>Column Details</h3>
          <table style={{ width: '100%', fontSize: '0.9rem' }}>
            <thead>
              <tr style={{ textAlign: 'left' }}>
                <th>Column</th>
                <th>Type</th>
                <th>Unique</th>
                <th>Missing</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(datasetInfo.dtypes).map(([col, dtype]) => (
                <tr key={col}>
                  <td><code>{col}</code></td>
                  <td>{dtype}</td>
                  <td>{datasetInfo.unique_counts[col]}</td>
                  <td>{datasetInfo.missing_values[col]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <h3 style={{ marginTop: '24px' }}>Sample Data</h3>
      <div className="table-container">
        <table>
          <thead>
            <tr>
              {datasetInfo.sample_rows[0] && Object.keys(datasetInfo.sample_rows[0]).map(col => (
                <th key={col}>{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {datasetInfo.sample_rows.map((row, i) => (
              <tr key={i}>
                {Object.values(row).map((val, j) => (
                  <td key={j}><code>{typeof val === 'string' && val.length > 50 ? val.substring(0, 50) + '...' : val}</code></td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 style={{ marginTop: '24px' }}>Limitations</h3>
      <ul style={{ marginLeft: '20px', lineHeight: '1.8', color: '#7f8c8d' }}>
        {datasetInfo.limitations.map((lim, i) => <li key={i}>{lim}</li>)}
      </ul>
    </div>
  );
}

export default DatasetInfo;