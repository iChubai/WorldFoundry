'use client';

import { Pause, Play } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';

type HeroVideo = {
  label: string;
  caption: string;
  src: string;
  poster: string;
};

/**
 * Product window for the hero: browser chrome around the showcase video.
 * Autoplays muted when motion is allowed; the poster stands in otherwise.
 */
export function HeroShowcase({ video }: { video: HeroVideo }) {
  const ref = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [canAutoplay, setCanAutoplay] = useState(false);

  useEffect(() => {
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const saveData = Boolean(
      (navigator as Navigator & { connection?: { saveData?: boolean } }).connection?.saveData,
    );
    setCanAutoplay(!reduce && !saveData);
  }, []);

  useEffect(() => {
    const element = ref.current;
    if (!element || !canAutoplay) return;
    element.play().then(
      () => setPlaying(true),
      () => setPlaying(false),
    );
  }, [canAutoplay]);

  function toggle() {
    const element = ref.current;
    if (!element) return;
    if (element.paused) {
      element.play().then(
        () => setPlaying(true),
        () => setPlaying(false),
      );
    } else {
      element.pause();
      setPlaying(false);
    }
  }

  return (
    <figure className="px-window">
      <div className="px-window-bar">
        <span className="px-window-dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span className="px-window-url">worldfoundry · {video.label.toLowerCase()}</span>
        <button
          type="button"
          className="px-window-play"
          onClick={toggle}
          aria-label={playing ? 'Pause showcase video' : 'Play showcase video'}
        >
          {playing ? (
            <Pause aria-hidden="true" size={13} strokeWidth={2} />
          ) : (
            <Play aria-hidden="true" size={13} strokeWidth={2} />
          )}
        </button>
      </div>
      <div className="px-window-media">
        <video
          ref={ref}
          src={canAutoplay ? video.src : undefined}
          poster={video.poster}
          muted
          loop
          playsInline
          preload="none"
          aria-label={video.caption}
        />
      </div>
      <figcaption className="px-window-caption">
        <span className="px-kicker-dot" aria-hidden="true" />
        <strong>{video.label}</strong>
        <span>{video.caption}</span>
      </figcaption>
    </figure>
  );
}
