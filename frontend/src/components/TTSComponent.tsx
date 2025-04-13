import React, { useState, useRef } from 'react';
import styles from './TTSComponent.module.css';

// Use actual voice IDs found in the hexgrad/Kokoro-82M repo
const voiceOptions = [
  // American Female Voices (af_ prefix)
  { value: 'af_alloy', label: 'Alloy (US Female)' }, 
  { value: 'af_aoede', label: 'Aoede (US Female)' },
  { value: 'af_bella', label: 'Bella (US Female)' },
  { value: 'af_heart', label: 'Heart (US Female)' },
  { value: 'af_jessica', label: 'Jessica (US Female)' },
  { value: 'af_kore', label: 'Kore (US Female)' },
  { value: 'af_nicole', label: 'Nicole (US Female)' },
  { value: 'af_nova', label: 'Nova (US Female)' },
  { value: 'af_river', label: 'River (US Female)' },
  { value: 'af_sarah', label: 'Sarah (US Female)' },
  { value: 'af_sky', label: 'Sky (US Female)' },
  // Other Voices (Examples)
  { value: 'am_echo', label: 'Echo (US Male)' },
  { value: 'bf_alice', label: 'Alice (UK Female)' },
  { value: 'bm_daniel', label: 'Daniel (UK Male)' },
  { value: 'jf_gongitsune', label: 'Gongitsune (JP Female)' },
  { value: 'zm_yunxi', label: 'Yunxi (CN Male)' },
];

const TTSComponent: React.FC = () => {
  const [text, setText] = useState<string>('');
  const [selectedVoice, setSelectedVoice] = useState<string>(voiceOptions[0].value);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  const handleConvert = async () => {
    if (!text.trim()) {
      setError('Please enter some text to convert.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setAudioUrl(null); // Clear previous audio

    try {
      // Assuming the backend runs on port 57103 as configured in tts.py
      // Adjust the URL if your backend setup is different
      const response = await fetch('http://localhost:57103/api/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text, voice: selectedVoice }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const audioBlob = await response.blob();
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);

    } catch (err) {
      console.error('TTS conversion failed:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
      setIsLoading(false);
    }
  };

  // Clean up the object URL when the component unmounts or audioUrl changes
  React.useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  return (
    <div className={styles.ttsContainer}>
      <h2>Text-to-Speech Converter (Kokoro TTS)</h2>
      <textarea
        className={styles.textArea}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text here..."
        rows={5}
        disabled={isLoading}
      />
      <div className={styles.controls}>
        <label htmlFor="voice-select">Select Voice:</label>
        <select
          id="voice-select"
          className={styles.select}
          value={selectedVoice}
          onChange={(e) => setSelectedVoice(e.target.value)}
          disabled={isLoading}
        >
          {voiceOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <button
          className={styles.convertButton}
          onClick={handleConvert}
          disabled={isLoading}
        >
          {isLoading ? 'Converting...' : 'Convert'}
        </button>
      </div>
      {error && <p className={styles.error}>Error: {error}</p>}
      {audioUrl && (
        <div className={styles.audioPlayerContainer}>
          <p>Conversion successful:</p>
          <audio ref={audioRef} controls src={audioUrl}>
            Your browser does not support the audio element.
          </audio>
        </div>
      )}
    </div>
  );
};

export default TTSComponent;
