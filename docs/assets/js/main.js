/* ============================================================
   RePrime — interactions
   ============================================================ */
(function () {
  'use strict';

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------------- nav, progress, active section ---------------- */
  const nav = $('#nav');
  const progress = $('.progress span');
  const navLinks = $$('.nav-links a');
  const sections = navLinks
    .map(a => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);

  function onScroll() {
    const y = window.scrollY;
    nav.classList.toggle('scrolled', y > 24);
    const h = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = (h > 0 ? Math.min(1, y / h) * 100 : 0) + '%';

    let current = null;
    for (const sec of sections) {
      if (sec.getBoundingClientRect().top <= 140) current = sec.id;
    }
    navLinks.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + current));
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* ---------------- reveal on scroll ---------------- */
  const revealEls = $$('.reveal');
  if ('IntersectionObserver' in window && !reduceMotion) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('is-in');
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.12, rootMargin: '0px 0px -6% 0px' });
    revealEls.forEach(el => io.observe(el));
  } else {
    revealEls.forEach(el => el.classList.add('is-in'));
  }

  /* ---------------- placement bench ---------------- */
  const bench = $('[data-bench]');
  if (bench) {
    const svg = $('#cloud');
    const dotsLayer = $('.plot-dots', svg);
    const gridLayer = $('.plot-grid', svg);
    const CX = 230, CY = 168;

    // grid
    for (let i = 1; i < 8; i++) {
      const x = (460 / 8) * i;
      const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      l.setAttribute('x1', x); l.setAttribute('x2', x);
      l.setAttribute('y1', 0); l.setAttribute('y2', 340);
      gridLayer.appendChild(l);
    }
    for (let i = 1; i < 6; i++) {
      const y = (340 / 6) * i;
      const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      l.setAttribute('x1', 0); l.setAttribute('x2', 460);
      l.setAttribute('y1', y); l.setAttribute('y2', y);
      gridLayer.appendChild(l);
    }

    // deterministic points
    function mulberry32(a) {
      return function () {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        let t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
      };
    }
    const rnd = mulberry32(2027);
    function gauss() {
      let u = 0, v = 0;
      while (!u) u = rnd();
      while (!v) v = rnd();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }

    const clusters = [
      { x: -82, y: -38, s: 27 },
      { x: 66, y: -54, s: 24 },
      { x: 16, y: 78, s: 30 }
    ];
    const pts = [];
    clusters.forEach((c, ci) => {
      for (let i = 0; i < 28; i++) {
        pts.push({ c: ci, ox: gauss() * c.s, oy: gauss() * c.s, r: 2.6 + rnd() * 1.7 });
      }
    });
    for (let i = 0; i < 12; i++) {
      const a = rnd() * Math.PI * 2;
      const rad = 112 + rnd() * 42;
      pts.push({ c: -1, ox: Math.cos(a) * rad, oy: Math.sin(a) * rad * 0.82, r: 2.4 + rnd() * 1.4 });
    }

    const circles = pts.map((p, i) => {
      const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      c.setAttribute('r', p.r.toFixed(2));
      c.style.transitionDelay = (i * 3) + 'ms';
      dotsLayer.appendChild(c);
      return c;
    });

    function layout(mode) {
      pts.forEach((p, i) => {
        let x, y;
        if (mode === 'sample') {
          const c = p.c >= 0 ? clusters[p.c] : clusters[0];
          const cx = c.x * 0.42, cy = c.y * 0.42;
          const s = 6.2;
          x = CX + cx + (p.c >= 0 ? p.ox / c.s * s : gauss() * s);
          y = CY + cy + (p.c >= 0 ? p.oy / c.s * s : gauss() * s);
        } else {
          const k = mode === 'pair' ? 1.1 : 1;
          x = CX + p.ox * k;
          y = CY + p.oy * k;
        }
        circles[i].style.transform = `translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`;
      });
    }

    const copy = {
      none: {
        status: 'Baseline · no auxiliary loss',
        verdict: 'Broad coverage, average fidelity.',
        desc: 'Post-training alone. The class is covered, but the samples are only as good as the representation the OT loss already sees.',
        recall: '0.452', fd: '14.71',
        foot: 'Same 2.5k-update protocol, Priming-B initialization.'
      },
      pair: {
        status: 'Paired path · instance-paired target',
        verdict: 'Better fidelity — and slightly better coverage.',
        desc: 'The loss sees the same image it is reconstructing. FD₆ falls and recall rises: the model gets better without becoming more typical.',
        recall: '0.469', fd: '12.89',
        foot: '20 / 20 paired-path comparisons improve FD₆ by 1–18%.'
      },
      sample: {
        status: 'Generative path · constructed target',
        verdict: 'Precision up. Coverage gone.',
        desc: 'Every target is invented, so the loss pulls all noise of a class toward one prototype. Precision stays high; the class’s variety disappears.',
        recall: '0.15–0.25', fd: '14.92–19.62',
        foot: '17 / 17 configurations collapse recall. λ down to 0.05 does not help.'
      }
    };

    const elStatus = $('[data-status]', bench);
    const elVerdict = $('[data-verdict]', bench);
    const elDesc = $('[data-desc]', bench);
    const elRecall = $('[data-recall]', bench);
    const elFd = $('[data-fd]', bench);
    const elFoot = $('[data-footnote]', bench);

    function animateNumber(el, to, isRange) {
      if (isRange) {
        el.animate(
          [{ opacity: 0, transform: 'translateY(6px)' }, { opacity: 1, transform: 'none' }],
          { duration: 320, easing: 'cubic-bezier(.22,.75,.22,1)' }
        );
        el.textContent = to;
        return;
      }
      const from = parseFloat(el.textContent.replace(/[^\d.]/g, '')) || 0;
      const target = parseFloat(to);
      const t0 = performance.now();
      const dur = 620;
      function step(t) {
        const k = Math.min(1, (t - t0) / dur);
        const e = 1 - Math.pow(1 - k, 3);
        el.textContent = (from + (target - from) * e).toFixed(3);
        if (k < 1) requestAnimationFrame(step);
        else el.textContent = to;
      }
      requestAnimationFrame(step);
    }

    function setMode(mode) {
      bench.dataset.mode = mode;
      $$('.segmented button', bench).forEach(b => {
        b.setAttribute('aria-selected', String(b.dataset.mode === mode));
      });
      const c = copy[mode];
      elStatus.textContent = c.status;
      elVerdict.textContent = c.verdict;
      elDesc.textContent = c.desc;
      elFoot.textContent = c.foot;
      animateNumber(elRecall, c.recall, c.recall.includes('–'));
      animateNumber(elFd, c.fd, c.fd.includes('–'));
      layout(mode);
    }

    $$('.segmented button', bench).forEach(btn => {
      btn.addEventListener('click', () => setMode(btn.dataset.mode));
      btn.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
          e.preventDefault();
          const btns = $$('.segmented button', bench);
          const i = btns.indexOf(btn);
          const next = btns[(i + (e.key === 'ArrowRight' ? 1 : btns.length - 1)) % btns.length];
          next.focus(); setMode(next.dataset.mode);
        }
      });
    });

    layout('none');
    // enable transitions only after the first paint
    requestAnimationFrame(() => {
      circles.forEach(c => { c.style.transition = 'transform .85s cubic-bezier(.22,.75,.22,1), fill .5s, opacity .5s'; });
    });
  }

  /* ---------------- bars ---------------- */
  function fillBars(scope) {
    $$('.corr-fill', scope).forEach(el => {
      const row = el.closest('.corr-row');
      const gain = parseFloat(row.dataset.gain || '0');
      el.style.width = (gain === 0 ? 0 : Math.max(4, gain)) + '%';
    });
    $$('.bar', scope).forEach(el => {
      const v = parseFloat(el.dataset.value);
      const min = parseFloat(el.dataset.min);
      const max = parseFloat(el.dataset.max);
      const pct = Math.max(6, Math.min(100, (max - v) / (max - min) * 100));
      el.style.setProperty('--h', pct + '%');
      el.style.height = pct + '%';
    });
  }
  const barScopes = $$('[data-bars]');
  if ('IntersectionObserver' in window && !reduceMotion) {
    const bo = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) { fillBars(e.target); bo.unobserve(e.target); }
      });
    }, { threshold: 0.25 });
    barScopes.forEach(s => bo.observe(s));
  } else {
    barScopes.forEach(fillBars);
  }

  /* ---------------- frontier chart ---------------- */
  const chart = $('#frontier');
  if (chart) {
    const NS = 'http://www.w3.org/2000/svg';
    const W = 920, H = 500;
    const X0 = 72, X1 = 888, Y0 = 42, Y1 = 428;
    const xmin = 100, xmax = 4200, ymin = 1.10, ymax = 2.22;
    const lx = v => X0 + (Math.log10(v) - Math.log10(xmin)) / (Math.log10(xmax) - Math.log10(xmin)) * (X1 - X0);
    const ly = v => Y1 - (v - ymin) / (ymax - ymin) * (Y1 - Y0);

    const data = [
      { name: 'pMF-H', x: 1968.5, y: 2.15, g: 'prior', note: 'prior work · 1 NFE', dx: 0, dy: -14, anchor: 'middle' },
      { name: 'iMF-XL/2', x: 3201.1, y: 1.74, g: 'prior', note: 'prior work · 1 NFE', dx: 0, dy: -14, anchor: 'middle' },
      { name: 'Drifting-L', x: 1476.4, y: 1.61, g: 'prior', note: 'prior work · 1280 ep', dx: 0, dy: -14, anchor: 'middle' },
      { name: 'WF-XL/2', x: 3744.4, y: 1.33, g: 'prior', note: 'prior work', dx: -12, dy: 4, anchor: 'end' },
      { name: 'Priming-B', x: 124.2, y: 1.373, g: 'base', note: '160 ep · 115M', dx: 12, dy: -8, anchor: 'start' },
      { name: 'Priming-L', x: 239.0, y: 1.174, g: 'base', note: '160 ep · 405M', dx: -12, dy: 2, anchor: 'end' },
      { name: 'RePrime-B', x: 157.3, y: 1.303, g: 'ours', note: '160 ep · 115M', dx: 0, dy: 26, anchor: 'middle' },
      { name: 'RePrime-B · block 8', x: 184.8, y: 1.266, g: 'ours', note: '200 ep · 115M', dx: 14, dy: 4, anchor: 'start' },
      { name: 'RePrime-J-B', x: 346.8, y: 1.224, g: 'ours', note: 'joint · 115M', dx: 12, dy: -14, anchor: 'start' },
      { name: 'RePrime-L', x: 272.1, y: 1.155, g: 'ours', note: '160 ep · 405M', dx: 14, dy: 4, anchor: 'start' }
    ];

    function el(tag, attrs, parent) {
      const n = document.createElementNS(NS, tag);
      for (const k in attrs) n.setAttribute(k, attrs[k]);
      if (parent) parent.appendChild(n);
      return n;
    }

    // grid + axes
    const gGrid = el('g', { class: 'chart-grid' }, chart);
    const gAxis = el('g', { class: 'chart-axis' }, chart);

    [1.2, 1.4, 1.6, 1.8, 2.0, 2.2].forEach(v => {
      const y = ly(v);
      el('line', { x1: X0, x2: X1, y1: y, y2: y }, gGrid);
      el('text', { x: X0 - 12, y: y + 3.5, 'text-anchor': 'end' }, gAxis).textContent = v.toFixed(1);
    });
    [100, 200, 500, 1000, 2000, 4000].forEach(v => {
      const x = lx(v);
      el('line', { x1: x, x2: x, y1: Y0, y2: Y1 }, gGrid);
      el('text', { x: x, y: Y1 + 22, 'text-anchor': 'middle' }, gAxis).textContent = v >= 1000 ? (v / 1000) + 'k' : v;
    });

    el('text', { x: (X0 + X1) / 2, y: H - 12, 'text-anchor': 'middle', class: 'chart-axis-label' }, chart)
      .textContent = 'Training compute · EFLOPs (log scale, teacher included)';
    const yl = el('text', { x: 18, y: (Y0 + Y1) / 2, 'text-anchor': 'middle', class: 'chart-axis-label', transform: `rotate(-90 18 ${(Y0 + Y1) / 2})` }, chart);
    yl.textContent = 'FID ↓';

    // better arrow
    const gAnn = el('g', { class: 'chart-ann' }, chart);
    el('text', { x: X0 + 8, y: Y0 + 18 }, gAnn).textContent = 'better';
    el('path', { d: `M${X0 + 52} ${Y0 + 14} l-14 14 m0 0 l6 -1 m-6 1 l1 -6`, fill: 'none' }, gAnn);

    // frontier line
    const frontier = el('path', {
      d: `M${lx(157.3)} ${ly(1.303)} C ${lx(200)} ${ly(1.27)}, ${lx(230)} ${ly(1.20)}, ${lx(272.1)} ${ly(1.155)}`,
      class: 'chart-frontier', fill: 'none'
    }, chart);

    // points
    const gPts = el('g', { class: 'chart-points' }, chart);
    const tip = $('#chart-tip');
    const wrap = $('#frontier-wrap');

    data.forEach((d, i) => {
      const x = lx(d.x), y = ly(d.y);
      const g = el('g', {
        class: 'chart-pt ' + d.g,
        tabindex: '0',
        role: 'button',
        'aria-label': `${d.name}: ${d.x} EFLOPs, FID ${d.y}`
      }, gPts);
      if (d.g === 'ours') el('circle', { cx: x, cy: y, r: 13, class: 'halo' }, g);
      el('circle', { cx: x, cy: y, r: d.g === 'ours' ? 6.5 : 5, class: 'dot' }, g);
      const t = el('text', { x: x + d.dx, y: y + d.dy, 'text-anchor': d.anchor }, g);
      t.textContent = d.name;

      function show() {
        const rect = chart.getBoundingClientRect();
        const k = rect.width / W;
        tip.innerHTML = `<div class="tip-name">${d.name}</div>
          <div class="tip-row"><span>compute</span><b>${d.x.toLocaleString()} EFLOPs</b></div>
          <div class="tip-row"><span>FID</span><b>${d.y.toFixed(3)}</b></div>
          <div class="tip-row"><span>setting</span><b>${d.note}</b></div>`;
        tip.style.left = (x * k) + 'px';
        tip.style.top = (y * k) + 'px';
        tip.classList.add('show');
        g.classList.add('active');
      }
      function hide() {
        tip.classList.remove('show');
        g.classList.remove('active');
      }
      g.addEventListener('mouseenter', show);
      g.addEventListener('mouseleave', hide);
      g.addEventListener('focus', show);
      g.addEventListener('blur', hide);
      g.addEventListener('click', show);
    });

    // animate frontier draw
    if (!reduceMotion) {
      const len = frontier.getTotalLength();
      frontier.style.strokeDasharray = len;
      frontier.style.strokeDashoffset = len;
      const fo = new IntersectionObserver((entries) => {
        entries.forEach(e => {
          if (e.isIntersecting) {
            frontier.style.transition = 'stroke-dashoffset 1.6s cubic-bezier(.22,.75,.22,1) .2s';
            frontier.style.strokeDashoffset = 0;
            fo.disconnect();
          }
        });
      }, { threshold: 0.3 });
      fo.observe(chart);
    }
  }

  /* ---------------- hover zoom + lightbox ---------------- */
  const lightbox = $('#lightbox');
  const lbImg = $('img', lightbox);
  const lbCap = $('figcaption', lightbox);
  let lastFocus = null;

  $$('[data-zoom]').forEach(img => {
    const wrap = img.closest('.zoom-wrap') || img.closest('.joint-plot') || img.parentElement;
    if (wrap && wrap.classList.contains('zoom-wrap')) {
      wrap.addEventListener('mousemove', (e) => {
        const r = wrap.getBoundingClientRect();
        wrap.style.setProperty('--zx', ((e.clientX - r.left) / r.width * 100) + '%');
        wrap.style.setProperty('--zy', ((e.clientY - r.top) / r.height * 100) + '%');
      });
    }
    img.setAttribute('tabindex', '0');
    img.setAttribute('role', 'button');
    img.setAttribute('aria-label', 'Enlarge figure: ' + (img.alt || 'figure'));
    img.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); img.click(); }
    });
    img.addEventListener('click', () => {
      lastFocus = document.activeElement;
      lbImg.src = img.currentSrc || img.src;
      lbImg.alt = img.alt || '';
      lbCap.textContent = img.dataset.caption || '';
      lightbox.classList.add('open');
      lightbox.setAttribute('aria-hidden', 'false');
      document.body.classList.add('no-scroll');
      const closeBtn = $('.lb-close', lightbox);
      requestAnimationFrame(() => closeBtn.focus({ preventScroll: true }));
    });
  });

  function closeLightbox() {
    lightbox.classList.remove('open');
    lightbox.setAttribute('aria-hidden', 'true');
    document.body.classList.remove('no-scroll');
    if (lastFocus) lastFocus.focus();
  }
  if (lightbox) {
    $('.lb-close', lightbox).addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', (e) => { if (e.target === lightbox) closeLightbox(); });
    document.addEventListener('keydown', (e) => {
      if (!lightbox.classList.contains('open')) return;
      if (e.key === 'Escape') closeLightbox();
      if (e.key === 'Tab') { e.preventDefault(); $('.lb-close', lightbox).focus(); }
    });
  }

  /* ---------------- video blocks ---------------- */
  $$('[data-video]').forEach(shell => {
    const video = $('video', shell);
    const btn = $('.video-play', shell);
    if (!video || !btn) return;

    btn.addEventListener('click', () => {
      const p = video.play();
      if (p && p.catch) p.catch(() => {});
    });
    video.addEventListener('play', () => shell.classList.add('playing'));
    video.addEventListener('ended', () => shell.classList.remove('playing'));

    // stop playback when the video leaves the viewport
    if ('IntersectionObserver' in window) {
      const vio = new IntersectionObserver((entries) => {
        entries.forEach(e => {
          if (!e.isIntersecting && !video.paused) video.pause();
        });
      }, { threshold: 0.05 });
      vio.observe(shell);
    }
  });

  /* ---------------- copy bibtex ---------------- */
  const copyBtn = $('.copy-btn');
  if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
      const text = $(copyBtn.dataset.copy).innerText;
      try {
        await navigator.clipboard.writeText(text);
      } catch (err) {
        const ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta); ta.select();
        document.execCommand('copy'); ta.remove();
      }
      const old = copyBtn.textContent;
      copyBtn.textContent = 'Copied ✓';
      setTimeout(() => { copyBtn.textContent = old; }, 1800);
    });
  }
  window.__reprimeReady = true;
})();
