import { useState, useEffect } from 'react';

function App() {
  const [file, setFile] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const backendUrl = '';

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setError('');
    setStatus('Uploading...');
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${backendUrl}/analyze-video`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      setTaskId(data.task_id);
      setStatus('pending');
    } catch (err) {
      setError(err.message);
      setStatus('');
    }
  };

  useEffect(() => {
    let intervalId;

    if (taskId && status === 'pending') {
      intervalId = setInterval(async () => {
        try {
          const response = await fetch(`${backendUrl}/get-analysis-result?task_id=${taskId}`);
          if (!response.ok) throw new Error('Polling failed');

          const data = await response.json();
          if (data.status === 'completed') {
            setStatus('completed');
            setResult({
              videoUrl: `${backendUrl}/${data.video_url}`,
              stats: data.verdict ? JSON.parse(data.verdict) : null
            });
            clearInterval(intervalId);
          } else if (data.status === 'failed') {
            setStatus('failed');
            setError('Analysis failed.');
            clearInterval(intervalId);
          }
        } catch (err) {
          console.error(err);
        }
      }, 3000);
    }

    return () => clearInterval(intervalId);
  }, [taskId, status]);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 p-8 font-sans">
      <div className="max-w-4xl mx-auto space-y-8">
        <header className="text-center">
          <title>Atletico Intelligence</title> 
          <h1 className="text-4xl font-bold tracking-tight text-white mb-2">Atletico Intelligence</h1>
          <p className="text-gray-400">Tactical football video analysis</p>
        </header>

        <section className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-xl max-w-lg mx-auto">
          <div className="flex flex-col items-center gap-4">
            <label className="w-full">
              <span className="sr-only">Choose video</span>
              <input
                type="file"
                accept="video/*"
                onChange={handleFileChange}
                className="block w-full text-sm text-gray-300
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-md file:border-0
                  file:text-sm file:font-semibold
                  file:bg-indigo-600 file:text-white
                  hover:file:bg-indigo-500
                  cursor-pointer"
              />
            </label>
            <button
              onClick={handleUpload}
              disabled={!file || status === 'Uploading...' || status === 'pending'}
              className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-600 border border-transparent rounded-md py-2 px-4 text-sm font-medium text-white transition-colors"
            >
              {status === 'Uploading...' ? 'Uploading...' : status === 'pending' ? 'Analyzing...' : 'Analyze Video'}
            </button>
            {error && <p className="text-red-400 text-sm">{error}</p>}
          </div>
        </section>

        {status === 'completed' && result && (
          <section className="space-y-6">
            <div className="bg-gray-800 rounded-xl overflow-hidden border border-gray-700 shadow-xl">
              <video
                controls
                autoPlay
                key={result.videoUrl}
                className="w-full max-h-[600px] object-contain bg-black"
              >
                <source src={result.videoUrl} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>

            {result.stats && (
              <div className="bg-gray-800 p-6 rounded-xl border border-gray-700 shadow-xl space-y-6">
                <h2 className="text-xl font-semibold border-b border-gray-700 pb-2">Analysis Verdict</h2>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-gray-900 p-4 rounded-lg">
                    <p className="text-gray-400 text-sm">Players</p>
                    <p className="text-2xl font-bold">{result.stats.players_detected}</p>
                  </div>
                  <div className="bg-gray-900 p-4 rounded-lg">
                    <p className="text-gray-400 text-sm">Passes</p>
                    <p className="text-2xl font-bold">{result.stats.passes_detected}</p>
                  </div>
                  <div className="bg-gray-900 p-4 rounded-lg">
                    <p className="text-gray-400 text-sm">Teammate Passes</p>
                    <p className="text-2xl font-bold">{result.stats.teammate_passes}</p>
                  </div>
                  <div className="bg-gray-900 p-4 rounded-lg">
                    <p className="text-gray-400 text-sm">Offsides</p>
                    <p className="text-2xl font-bold">{result.stats.offside_offences}</p>
                  </div>
                </div>

                {result.stats.decisions && result.stats.decisions.length > 0 && (
                  <div className="mt-6">
                    <h3 className="text-lg font-medium mb-3">Decisions</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm text-left">
                        <thead className="text-xs uppercase bg-gray-900 text-gray-400">
                          <tr>
                            <th className="px-4 py-3 rounded-tl-lg">Frame</th>
                            <th className="px-4 py-3">Passer</th>
                            <th className="px-4 py-3">Receiver</th>
                            <th className="px-4 py-3">Beyond Line</th>
                            <th className="px-4 py-3 rounded-tr-lg">Verdict</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.stats.decisions.map((dec, idx) => (
                            <tr key={idx} className="border-b border-gray-800">
                              <td className="px-4 py-3">{dec.kick_frame}</td>
                              <td className="px-4 py-3">{dec.passer}</td>
                              <td className="px-4 py-3">{dec.receiver}</td>
                              <td className="px-4 py-3">{dec.attackers_beyond_line}</td>
                              <td className="px-4 py-3">
                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${dec.verdict === 'onside' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
                                  {dec.verdict.toUpperCase()}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
}

export default App;
