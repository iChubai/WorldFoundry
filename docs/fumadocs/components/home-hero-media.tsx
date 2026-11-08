'use client';

import { Play } from 'lucide-react';
import { type ReactNode, useEffect, useRef, useState } from 'react';

import { homeHeroVideo } from '@/lib/home-hero-video';

type ConnectionLike = EventTarget & {
  saveData?: boolean;
};

export function HomeHeroMedia({ children }: { children?: ReactNode }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [motionAllowed, setMotionAllowed] = useState(false);
  const [videoAllowed, setVideoAllowed] = useState(false);
  const [pageVisible, setPageVisible] = useState(true);
  const [heroVisible, setHeroVisible] = useState(true);
  const [videoReady, setVideoReady] = useState(false);
  const [playbackRequested, setPlaybackRequested] = useState(false);
  const [playing, setPlaying] = useState(false);
  const mountedRef = useRef(false);

  const showVideo = (videoAllowed && motionAllowed) || playbackRequested;

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const connection = (navigator as Navigator & { connection?: ConnectionLike }).connection;
    let videoTimer = 0;
    let saveData = Boolean(connection?.saveData);

    const updatePreference = () => {
      window.clearTimeout(videoTimer);
      const allowsMotion = !preference.matches;
      setMotionAllowed(allowsMotion);
      if (allowsMotion && !saveData) {
        videoTimer = window.setTimeout(() => setVideoAllowed(true), 900);
      } else {
        setVideoAllowed(false);
        setVideoReady(false);
      }
    };

    const updateConnection = () => {
      const nextSaveData = Boolean(connection?.saveData);
      if (nextSaveData === saveData) return;
      saveData = nextSaveData;
      updatePreference();
    };

    updatePreference();
    preference.addEventListener('change', updatePreference);
    connection?.addEventListener('change', updateConnection);

    return () => {
      window.clearTimeout(videoTimer);
      preference.removeEventListener('change', updatePreference);
      connection?.removeEventListener('change', updateConnection);
    };
  }, []);

  useEffect(() => {
    const updateVisibility = () => setPageVisible(document.visibilityState !== 'hidden');
    updateVisibility();
    document.addEventListener('visibilitychange', updateVisibility);
    return () => document.removeEventListener('visibilitychange', updateVisibility);
  }, []);

  useEffect(() => {
    const root = document.querySelector('.wf-home-shell');
    root?.classList.toggle('wf-home-motion-paused', !motionAllowed);
    return () => root?.classList.remove('wf-home-motion-paused');
  }, [motionAllowed]);

  useEffect(() => {
    const root = document.querySelector('.wf-home-hero-carousel-shell');
    if (!root || typeof IntersectionObserver === 'undefined') return;

    const observer = new IntersectionObserver(
      ([entry]) => setHeroVisible(entry?.isIntersecting ?? true),
      { threshold: 0.12 },
    );
    observer.observe(root);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    if (video.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
      setVideoReady(true);
    }

    if (!pageVisible || !heroVisible || !showVideo) {
      video.pause();
      return;
    }

    video.muted = true;
    video.defaultMuted = true;
    void video.play().catch(() => {
      if (mountedRef.current) setPlaying(false);
    });
  }, [heroVisible, motionAllowed, pageVisible, showVideo]);

  return (
    <div className="wf-home-hero-carousel-shell" aria-label="World model showcase">
      <div className="wf-home-hero-carousel">
        <div className="wf-home-hero-carousel-viewport">
          <div className="wf-home-hero-carousel-track">
            <article className="wf-home-hero-card is-active">
              <div className="wf-home-hero-card-frame">
                <div className="wf-home-hero-card-media">
                  <img className="wf-home-hero-card-poster" src={homeHeroVideo.poster} alt="" aria-hidden="true" />
                  <video
                    ref={videoRef}
                    className={`wf-home-hero-card-video${videoReady ? ' is-ready' : ''}`}
                    src={showVideo ? homeHeroVideo.src : undefined}
                    poster={homeHeroVideo.poster}
                    autoPlay={showVideo}
                    loop
                    muted
                    playsInline
                    preload={showVideo ? 'auto' : 'none'}
                    aria-hidden="true"
                    tabIndex={-1}
                    onCanPlay={() => {
                      if (mountedRef.current) setVideoReady(true);
                    }}
                    onPlaying={() => setPlaying(true)}
                    onPause={() => setPlaying(false)}
                    onError={() => {
                      setPlaying(false);
                      setVideoReady(false);
                    }}
                  />
                </div>
                {children ? <div className="wf-home-hero-card-overlay">{children}</div> : null}
                {playing ? null : (
                  <button
                    type="button"
                    className="wf-home-hero-play"
                    aria-label="Play background video"
                    onClick={() => {
                      const video = videoRef.current;
                      if (!video) return;
                      video.muted = true;
                      video.defaultMuted = true;
                      if (!video.getAttribute('src')) video.src = homeHeroVideo.src;
                      setPlaybackRequested(true);
                      void video.play().catch(() => setPlaying(false));
                    }}
                  >
                    <Play aria-hidden="true" size={16} />
                    Play video
                  </button>
                )}
              </div>
            </article>
          </div>
        </div>

        <div className="wf-home-hero-carousel-caption" aria-live="polite">
          <strong>{homeHeroVideo.label}</strong>
          <span>{homeHeroVideo.caption}</span>
        </div>
      </div>
    </div>
  );
}
