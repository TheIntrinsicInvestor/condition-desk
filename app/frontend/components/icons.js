// Authored icons, one 1.5px stroke throughout. No glyph or emoji stand-ins.

const base = {
  width: 16,
  height: 16,
  viewBox: "0 0 16 16",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.5,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
};

export function IconUpload(p) {
  return (
    <svg {...base} {...p}>
      <path d="M8 10.5V2.5" />
      <path d="M5 5.5 8 2.5l3 3" />
      <path d="M2.5 10.5v2a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-2" />
    </svg>
  );
}

export function IconDownload(p) {
  return (
    <svg {...base} {...p}>
      <path d="M8 2.5v8" />
      <path d="m5 7.5 3 3 3-3" />
      <path d="M2.5 11.5v1a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-1" />
    </svg>
  );
}

export function IconAlert(p) {
  return (
    <svg {...base} {...p}>
      <path d="M8 2.6 1.9 13.1h12.2L8 2.6Z" />
      <path d="M8 6.6v3" />
      <path d="M8 11.4h.01" />
    </svg>
  );
}

export function IconCheck(p) {
  return (
    <svg {...base} {...p}>
      <circle cx="8" cy="8" r="5.6" />
      <path d="m5.7 8.1 1.6 1.6 3-3.4" />
    </svg>
  );
}

export function IconFault(p) {
  return (
    <svg {...base} {...p}>
      <circle cx="8" cy="8" r="5.6" />
      <path d="M8 5.1v3.5" />
      <path d="M8 10.7h.01" />
    </svg>
  );
}

// Subsystem marks, each a literal picture of what it measures.

export function IconShm(p) {
  return (
    <svg {...base} {...p}>
      <path d="M1.8 8.6c1-3.2 1.7 3.4 2.7.2.9-3 1.6 4.2 2.6 1.2 1-3.1 1.7 2.8 2.6.4 1-2.6 1.7 2.4 2.6.6" />
    </svg>
  );
}

export function IconAcv(p) {
  return (
    <svg {...base} {...p}>
      <path d="M8 2v12" />
      <path d="M2.8 5 13.2 11" />
      <path d="M13.2 5 2.8 11" />
    </svg>
  );
}

export function IconDoor(p) {
  return (
    <svg {...base} {...p}>
      <path d="M2.4 2.6v10.8" />
      <path d="M13.6 2.6v10.8" />
      <path d="M6.2 4.2v7.6" />
      <path d="M9.8 4.2v7.6" />
    </svg>
  );
}

export function IconRail(p) {
  return (
    <svg {...base} {...p}>
      <path d="M5.6 2.4 4.2 13.6" />
      <path d="M10.4 2.4l1.4 11.2" />
      <path d="M3.1 6.2h9.8" />
      <path d="M2.8 10.2h10.4" />
    </svg>
  );
}

export const SUBSYSTEM_ICONS = {
  shm: IconShm,
  acv: IconAcv,
  door: IconDoor,
  rail: IconRail,
};
